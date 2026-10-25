#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <mma.h>
#include <cmath>
#include <type_traits>
#include <vector>

// Teaching implementations, independent of vLLM's implementation.
// All tensor strides and addresses are in elements, never bytes.
struct S3 { int64_t a, b, c; };
static S3 s3(const at::Tensor& t) {
  return {t.stride(0), t.stride(1), t.stride(2)};
}
struct CacheDesc {
  int64_t s[5];
  int block_size, x;
  bool head_major, key;
};
static CacheDesc cache_desc(const at::Tensor& t, bool head_major, bool key,
                            int block_size, int x) {
  CacheDesc c = {{0, 0, 0, 0, 0}, block_size, x, head_major, key};
  for (int i = 0; i < t.dim(); ++i) c.s[i] = t.stride(i);
  return c;
}
__device__ int64_t cache_offset(CacheDesc c, int64_t slot, int h, int d) {
  int64_t block = slot / c.block_size, off = slot % c.block_size;
  if (!c.head_major) return block*c.s[0] + off*c.s[1] + h*c.s[2] + d*c.s[3];
  if (c.key) return block*c.s[0] + h*c.s[1] + (d/c.x)*c.s[2]
                   + off*c.s[3] + (d%c.x)*c.s[4];
  return block*c.s[0] + h*c.s[1] + d*c.s[2] + off*c.s[3];
}
__device__ float warp_sum(float x) {
  for (int d = 16; d; d >>= 1) x += __shfl_down_sync(0xffffffff, x, d);
  return __shfl_sync(0xffffffff, x, 0);
}
__device__ float warp_max(float x) {
  for (int d = 16; d; d >>= 1) x = fmaxf(x, __shfl_down_sync(0xffffffff, x, d));
  return __shfl_sync(0xffffffff, x, 0);
}
template<class T> __device__ float as_float(T x) { return float(x); }
template<class T> __device__ T from_float(float x) { return T(x); }
template<> __device__ float as_float(__half x) { return __half2float(x); }
template<> __device__ float as_float(__nv_bfloat16 x) { return __bfloat162float(x); }
template<> __device__ __half from_float(float x) { return __float2half_rn(x); }
template<> __device__ __nv_bfloat16 from_float(float x) { return __float2bfloat16_rn(x); }
template<class T> struct alignas(sizeof(T)*4) Pack4 { T x[4]; };

#define INPUT_DISPATCH(tensor, ...) \
  switch ((tensor).scalar_type()) { \
    case at::kFloat: { using scalar_t = float; __VA_ARGS__; break; } \
    case at::kHalf: { using scalar_t = __half; __VA_ARGS__; break; } \
    case at::kBFloat16: { using scalar_t = __nv_bfloat16; __VA_ARGS__; break; } \
    default: TORCH_CHECK(false, "unsupported input dtype"); \
  }
#define STATE_DISPATCH(tensor, ...) \
  switch ((tensor).scalar_type()) { \
    case at::kFloat: { using state_t = float; __VA_ARGS__; break; } \
    case at::kHalf: { using state_t = __half; __VA_ARGS__; break; } \
    case at::kBFloat16: { using state_t = __nv_bfloat16; __VA_ARGS__; break; } \
    default: TORCH_CHECK(false, "unsupported state dtype"); \
  }

template<class T, class Dst, bool Vector>
__global__ void cache_kernel(const T* k, const T* v, Dst* kc, Dst* vc,
    const int64_t* slots, S3 ks, S3 vs, CacheDesc kd, CacheDesc vd,
    int64_t tokens, int heads, int dim, int64_t capacity, float sk, float sv) {
  int64_t t, pos;
  if constexpr (Vector) {
    t = blockIdx.x;
    pos = (int64_t(blockIdx.y)*blockDim.x + threadIdx.x)*4;
  } else {
    int64_t z = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    t = z / (heads*dim);
    pos = z % (heads*dim);
  }
  if (t >= tokens || pos >= heads*dim) return;
  int64_t slot = slots[t];
  if (slot < 0 || slot >= capacity) return;
  int h = pos/dim, d = pos%dim;
  int64_t k_idx, v_idx;
  k_idx = t * ks.a + h * ks.b + d;
  v_idx = t * vs.a + h * vs.b + d;
  if constexpr (Vector) {
    auto kk = *reinterpret_cast<const Pack4<T>*>(k + k_idx);
    auto vv = *reinterpret_cast<const Pack4<T>*>(v + v_idx);
    Pack4<Dst> ko, vo;
    #pragma unroll
    for (int i=0; i<4; ++i) {
      ko.x[i] = from_float<Dst>(as_float(kk.x[i])/sk);
      vo.x[i] = from_float<Dst>(as_float(vv.x[i])/sv);
    }
    *reinterpret_cast<Pack4<Dst>*>(kc + cache_offset(kd,slot,h,d)) = ko;
    *reinterpret_cast<Pack4<Dst>*>(vc + cache_offset(vd,slot,h,d)) = vo;
  } else {
    kc[cache_offset(kd,slot,h,d)] = from_float<Dst>(as_float(k[t*ks.a+h*ks.b+d])/sk);
    vc[cache_offset(vd,slot,h,d)] = from_float<Dst>(as_float(v[t*vs.a+h*vs.b+d])/sv);
  }
}
template<class T, class Dst>
static void launch_cache(at::Tensor k, at::Tensor v, at::Tensor kc, at::Tensor vc,
    at::Tensor slots, float sk, float sv, bool optimized, cudaStream_t stream) {
  bool hm = kc.dim()==5;
  int b = hm ? kc.size(3) : kc.size(1), x = hm ? kc.size(4) : 1;
  auto kd=cache_desc(kc,hm,true,b,x), vd=cache_desc(vc,hm,false,b,x);
  int64_t t=k.size(0); int h=k.size(1), d=k.size(2);
  bool vec=optimized && !hm && d%4==0 && k.stride(1)==d && v.stride(1)==d
    && reinterpret_cast<uintptr_t>(k.data_ptr())%alignof(Pack4<T>)==0
    && reinterpret_cast<uintptr_t>(v.data_ptr())%alignof(Pack4<T>)==0
    && reinterpret_cast<uintptr_t>(kc.data_ptr())%alignof(Pack4<Dst>)==0
    && reinterpret_cast<uintptr_t>(vc.data_ptr())%alignof(Pack4<Dst>)==0
    && k.stride(0)%4==0 && v.stride(0)%4==0
    && kc.stride(3)==1 && vc.stride(3)==1
    && kc.stride(2)%4==0 && vc.stride(2)%4==0
    && kc.stride(1)%4==0 && vc.stride(1)%4==0
    && kc.stride(0)%4==0 && vc.stride(0)%4==0;
  if (!t) return;
  if (vec) {
    cache_kernel<T,Dst,true><<<dim3(t,(h*d+511)/512),128,0,stream>>>(
      (const T*)k.data_ptr(),(const T*)v.data_ptr(),(Dst*)kc.data_ptr(),(Dst*)vc.data_ptr(),
      slots.data_ptr<int64_t>(),s3(k),s3(v),kd,vd,t,h,d,kc.size(0)*b,sk,sv);
  } else {
    cache_kernel<T,Dst,false><<<(t*h*d+127)/128,128,0,stream>>>(
      (const T*)k.data_ptr(),(const T*)v.data_ptr(),(Dst*)kc.data_ptr(),(Dst*)vc.data_ptr(),
      slots.data_ptr<int64_t>(),s3(k),s3(v),kd,vd,t,h,d,kc.size(0)*b,sk,sv);
  }
}
void cache_cuda(at::Tensor k, at::Tensor v, at::Tensor kc, at::Tensor vc,
                at::Tensor slots, double sk, double sv, bool opt) {
  c10::cuda::CUDAGuard guard(k.device());
  auto stream=at::cuda::getCurrentCUDAStream(k.get_device());
  INPUT_DISPATCH(k, {
    if (kc.scalar_type()==at::kFloat8_e4m3fn)
      launch_cache<scalar_t,__nv_fp8_e4m3>(k,v,kc,vc,slots,sk,sv,opt,stream);
    else launch_cache<scalar_t,scalar_t>(k,v,kc,vc,slots,1,1,opt,stream);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Baseline reduces serially in one thread; optimized uses a warp per head.
template<class T, class Dst, bool Parallel>
__global__ void dynamic_kernel(const T* k, const T* v, Dst* kc, Dst* vc,
    float* sc_k, float* sc_v, const int64_t* slots, S3 ks, S3 vs,
    CacheDesc kd, CacheDesc vd, S3 sks, S3 svs, int heads, int dk, int dv,
    int64_t capacity, float qmax, float qmin) {
  int t=blockIdx.x, h=blockIdx.y, lane=threadIdx.x;
  int64_t slot=slots[t];
  if (slot<0 || slot>=capacity) return;
  constexpr int step=Parallel ? 32 : 1;
  float km=0, vm=0;
  for (int d=lane; d<dk; d+=step) km=fmaxf(km,fabsf(as_float(k[t*ks.a+h*ks.b+d])));
  for (int d=lane; d<dv; d+=step) vm=fmaxf(vm,fabsf(as_float(v[t*vs.a+h*vs.b+d])));
  if constexpr (Parallel) { km=warp_max(km); vm=warp_max(vm); }
  float sk=fmaxf(km/qmax,1e-6f), sv=fmaxf(vm/qmax,1e-6f);
  int64_t block=slot/kd.block_size, off=slot%kd.block_size;
  if (lane==0) {
    sc_k[block*sks.a+off*sks.b+h*sks.c]=sk;
    sc_v[block*svs.a+off*svs.b+h*svs.c]=sv;
  }
  for (int d=lane; d<dk; d+=step) {
    float x=as_float(k[t*ks.a+h*ks.b+d])*(1.0f/sk);
    if constexpr (std::is_same<Dst,int8_t>::value) x=truncf(x+copysignf(0.5f,x));
    kc[cache_offset(kd,slot,h,d)]=from_float<Dst>(fminf(qmax,fmaxf(qmin,x)));
  }
  for (int d=lane; d<dv; d+=step) {
    float x=as_float(v[t*vs.a+h*vs.b+d])*(1.0f/sv);
    if constexpr (std::is_same<Dst,int8_t>::value) x=truncf(x+copysignf(0.5f,x));
    vc[cache_offset(vd,slot,h,d)]=from_float<Dst>(fminf(qmax,fmaxf(qmin,x)));
  }
}
template<class T, class Dst>
static void launch_dynamic(at::Tensor k, at::Tensor v, at::Tensor kc, at::Tensor vc,
    at::Tensor sk, at::Tensor sv, at::Tensor slots, bool opt, cudaStream_t stream) {
  auto kd=cache_desc(kc,false,true,kc.size(1),1);
  auto vd=cache_desc(vc,false,false,kc.size(1),1);
  float max=std::is_same<Dst,int8_t>::value ? 127.f : 448.f;
  float min=std::is_same<Dst,int8_t>::value ? -128.f : -448.f;
  if (!k.size(0)) return;
  #define DYNAMIC_ARGS (const T*)k.data_ptr(),(const T*)v.data_ptr(),(Dst*)kc.data_ptr(), \
    (Dst*)vc.data_ptr(),sk.data_ptr<float>(),sv.data_ptr<float>(),slots.data_ptr<int64_t>(), \
    s3(k),s3(v),kd,vd,s3(sk),s3(sv),int(k.size(1)),int(k.size(2)),int(v.size(2)),kc.size(0)*kc.size(1),max,min
  if (opt) dynamic_kernel<T,Dst,true><<<dim3(k.size(0),k.size(1)),32,0,stream>>>(DYNAMIC_ARGS);
  else dynamic_kernel<T,Dst,false><<<dim3(k.size(0),k.size(1)),1,0,stream>>>(DYNAMIC_ARGS);
  #undef DYNAMIC_ARGS
}
void dynamic_cache_cuda(at::Tensor k, at::Tensor v, at::Tensor kc, at::Tensor vc,
    at::Tensor sk, at::Tensor sv, at::Tensor slots, bool opt) {
  c10::cuda::CUDAGuard guard(k.device());
  auto stream=at::cuda::getCurrentCUDAStream(k.get_device());
  INPUT_DISPATCH(k, {
    if (kc.scalar_type()==at::kChar)
      launch_dynamic<scalar_t,int8_t>(k,v,kc,vc,sk,sv,slots,opt,stream);
    else launch_dynamic<scalar_t,__nv_fp8_e4m3>(k,v,kc,vc,sk,sv,slots,opt,stream);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// One CTA per query. This intentionally repeats K/V loads between queries.
template<class T>
__global__ void attention_naive(const T* q,const T* k,const T* v,T* out,
    const int32_t* cu,S3 qs,S3 ks,S3 vs,int heads,int dim,float scale) {
  int beg=cu[blockIdx.z], end=cu[blockIdx.z+1], row=blockIdx.x, h=blockIdx.y;
  if (row>=end-beg) return;
  int lane=threadIdx.x%32, warp=threadIdx.x/32;
  __shared__ float partial[4], weights[3];
  float acc[2]={0,0}, m=-INFINITY, l=0;
  for (int j=beg;j<end;++j) {
    float dot=0;
    for(int d=threadIdx.x;d<dim;d+=128)
      dot+=as_float(q[(beg+row)*qs.a+h*qs.b+d])*as_float(k[j*ks.a+h*ks.b+d]);
    dot=warp_sum(dot);
    if(lane==0) partial[warp]=dot;
    __syncthreads();
    if(threadIdx.x==0) {
      dot=(partial[0]+partial[1]+partial[2]+partial[3])*scale;
      float next=fmaxf(m,dot), alpha=expf(m-next), p=expf(dot-next);
      l=alpha*l+p; m=next;
      weights[0]=alpha; weights[1]=p; weights[2]=l;
    }
    __syncthreads();
    for(int d=threadIdx.x,u=0;d<dim;d+=128,++u)
      acc[u]=weights[0]*acc[u]+weights[1]*as_float(v[j*vs.a+h*vs.b+d]);
    __syncthreads();
  }
  for(int d=threadIdx.x,u=0;d<dim;d+=128,++u)
    out[((int64_t(beg)+row)*heads+h)*dim+d]=from_float<T>(acc[u]/weights[2]);
}

template<class T,int DP>
__device__ void copy_kv(T* kd,T* vd,const T* k,const T* v,S3 ks,S3 vs,
    int start,int end,int h,int dim,int tid) {
  for(int z=tid;z<16*DP;z+=64) {
    int r=z/DP,d=z%DP;
    kd[z]=(start+r<end && d<dim) ? k[(start+r)*ks.a+h*ks.b+d] : from_float<T>(0);
    vd[z]=(start+r<end && d<dim) ? v[(start+r)*vs.a+h*vs.b+d] : from_float<T>(0);
  }
}
template<class T,int DP>
__device__ void copy_kv_async(T* kd,T* vd,const T* k,const T* v,S3 ks,S3 vs,
    int start,int end,int h,int dim,int tid) {
  #if __CUDA_ARCH__ >= 800
  for(int z=tid;z<16*DP/8;z+=64) {
    int r=z/(DP/8),d=(z%(DP/8))*8;
    bool valid=start+r<end && d<dim;
    const T* kp=valid ? k+(start+r)*ks.a+h*ks.b+d : k;
    const T* vp=valid ? v+(start+r)*vs.a+h*vs.b+d : v;
    unsigned ka=__cvta_generic_to_shared(kd+z*8), va=__cvta_generic_to_shared(vd+z*8);
    int bytes=valid ? 16 : 0;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;" :: "r"(ka),"l"(kp),"r"(bytes));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;" :: "r"(va),"l"(vp),"r"(bytes));
  }
  asm volatile("cp.async.commit_group;");
  #endif
}

// Q is partitioned between warps: each warp owns 16 complete output rows.
// K/V tiles are shared; no cross-warp partial-output reduction is needed.
template<class T,int DP,bool TensorCore,bool Async>
__global__ void attention_tiled(const T* q,const T* k,const T* v,T* out,
    const int32_t* cu,S3 qs,S3 ks,S3 vs,int heads,int dim,float scale) {
  namespace wmma=nvcuda::wmma;
  extern __shared__ __align__(32) unsigned char storage[];
  T* qt=(T*)storage;
  T* kt=qt+32*DP;
  T* vt=kt+2*16*DP;
  T* prob=vt+2*16*DP;
  float* scores=(float*)(prob+32*16);
  float* numer=scores+32*16;
  float* maxima=numer+32*DP;
  float* denom=maxima+32;
  float* alpha=denom+32;
  int tid=threadIdx.x,warp=tid/32,lane=tid%32,h=blockIdx.y;
  int beg=cu[blockIdx.z],end=cu[blockIdx.z+1],qbase=blockIdx.x*32;
  if(qbase>=end-beg) return;
  for(int z=tid;z<32*DP;z+=64) {
    int r=z/DP,d=z%DP;
    qt[z]=(beg+qbase+r<end && d<dim) ? q[(beg+qbase+r)*qs.a+h*qs.b+d] : from_float<T>(0);
    numer[z]=0;
  }
  if(tid<32) {maxima[tid]=-INFINITY;denom[tid]=0;}
  if constexpr(Async) copy_kv_async<T,DP>(kt,vt,k,v,ks,vs,beg,end,h,dim,tid);
  __syncthreads();
  for(int start=beg,step=0;start<end;start+=16,++step) {
    int buf=step%2;
    T* curk=kt+buf*16*DP; T* curv=vt+buf*16*DP;
    if constexpr(Async) {
      #if __CUDA_ARCH__ >= 800
      asm volatile("cp.async.wait_group 0;");
      #endif
      __syncthreads();
      if(start+16<end)
        copy_kv_async<T,DP>(kt+(1-buf)*16*DP,vt+(1-buf)*16*DP,k,v,ks,vs,start+16,end,h,dim,tid);
    } else {
      copy_kv<T,DP>(curk,curv,k,v,ks,vs,start,end,h,dim,tid);
      __syncthreads();
    }
    if constexpr(TensorCore) {
      #if __CUDA_ARCH__ >= 800
      wmma::fragment<wmma::matrix_a,16,16,16,T,wmma::row_major> qa;
      wmma::fragment<wmma::matrix_b,16,16,16,T,wmma::col_major> kb;
      wmma::fragment<wmma::accumulator,16,16,16,float> score;
      wmma::fill_fragment(score,0.f);
      for(int d=0;d<DP;d+=16) {
        wmma::load_matrix_sync(qa,qt+warp*16*DP+d,DP);
        wmma::load_matrix_sync(kb,curk+d,DP);
        wmma::mma_sync(score,qa,kb,score);
      }
      wmma::store_matrix_sync(scores+warp*16*16,score,16,wmma::mem_row_major);
      #endif
    } else {
      for(int z=tid;z<32*16;z+=64) {
        int r=z/16,j=z%16; float x=0;
        for(int d=0;d<DP;++d) x=fmaf(as_float(qt[r*DP+d]),as_float(curk[j*DP+d]),x);
        scores[z]=x;
      }
    }
    __syncthreads();
    if(lane<16) {
      int r=warp*16+lane;
      bool valid=beg+qbase+r<end;
      float m=-INFINITY;
      for(int j=0;j<16;++j) {
        float x=(valid && start+j<end) ? scores[r*16+j]*scale : -INFINITY;
        scores[r*16+j]=x; m=fmaxf(m,x);
      }
      float next=valid ? fmaxf(maxima[r],m) : 0;
      float a=valid ? expf(maxima[r]-next) : 0, sum=0;
      for(int j=0;j<16;++j) {
        float p=valid ? expf(scores[r*16+j]-next) : 0;
        prob[r*16+j]=from_float<T>(p); sum+=p;
      }
      alpha[r]=a; denom[r]=denom[r]*a+sum; maxima[r]=next;
    }
    __syncthreads();
    if constexpr(TensorCore) {
      for(int z=tid;z<32*DP;z+=64) numer[z]*=alpha[z/DP];
      __syncthreads();
      #if __CUDA_ARCH__ >= 800
      wmma::fragment<wmma::matrix_a,16,16,16,T,wmma::row_major> p;
      wmma::fragment<wmma::matrix_b,16,16,16,T,wmma::row_major> val;
      wmma::fragment<wmma::accumulator,16,16,16,float> o;
      wmma::load_matrix_sync(p,prob+warp*16*16,16);
      for(int d=0;d<DP;d+=16) {
        wmma::load_matrix_sync(val,curv+d,DP);
        wmma::load_matrix_sync(o,numer+warp*16*DP+d,DP,wmma::mem_row_major);
        wmma::mma_sync(o,p,val,o);
        wmma::store_matrix_sync(numer+warp*16*DP+d,o,DP,wmma::mem_row_major);
      }
      #endif
    } else {
      for(int z=tid;z<32*DP;z+=64) {
        int r=z/DP,d=z%DP; float x=numer[z]*alpha[r];
        for(int j=0;j<16;++j) x=fmaf(as_float(prob[r*16+j]),as_float(curv[j*DP+d]),x);
        numer[z]=x;
      }
    }
    __syncthreads();
  }
  for(int z=tid;z<32*DP;z+=64) {
    int r=z/DP,d=z%DP;
    if(beg+qbase+r<end && d<dim)
      out[((int64_t(beg)+qbase+r)*heads+h)*dim+d]=from_float<T>(numer[z]/denom[r]);
  }
}
template<class T,int DP,bool TC,bool ASYNC>
static void launch_tiled(at::Tensor q,at::Tensor k,at::Tensor v,at::Tensor out,
    at::Tensor cu,int max_seq,float scale,cudaStream_t stream) {
  size_t shared=sizeof(T)*(32*DP+4*16*DP+32*16)
    +sizeof(float)*(32*16+32*DP+3*32);
  auto kernel=attention_tiled<T,DP,TC,ASYNC>;
  if(shared>48*1024)
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,shared));
  kernel<<<dim3((max_seq+31)/32,q.size(1),cu.numel()-1),64,shared,stream>>>(
    (const T*)q.data_ptr(),(const T*)k.data_ptr(),(const T*)v.data_ptr(),(T*)out.data_ptr(),
    cu.data_ptr<int32_t>(),s3(q),s3(k),s3(v),q.size(1),q.size(2),scale);
}
template<class T,int DP>
static void attention_select(at::Tensor q,at::Tensor k,at::Tensor v,at::Tensor out,
    at::Tensor cu,int max_seq,float scale,int impl,cudaStream_t stream) {
  if constexpr(std::is_same<T,float>::value) {
    launch_tiled<T,DP,false,false>(q,k,v,out,cu,max_seq,scale,stream);
  } else {
    if(impl==1) launch_tiled<T,DP,false,false>(q,k,v,out,cu,max_seq,scale,stream);
    else if(impl==2) launch_tiled<T,DP,true,false>(q,k,v,out,cu,max_seq,scale,stream);
    else launch_tiled<T,DP,true,true>(q,k,v,out,cu,max_seq,scale,stream);
  }
}
at::Tensor attention_cuda(at::Tensor q,at::Tensor k,at::Tensor v,at::Tensor cu,
                          int64_t max_seq,double scale,int64_t impl) {
  c10::cuda::CUDAGuard guard(q.device());
  auto stream=at::cuda::getCurrentCUDAStream(q.get_device());
  auto out=at::empty(q.sizes(),q.options());
  if(!q.numel() || !max_seq) return out;
  INPUT_DISPATCH(q, {
    if(impl==0) attention_naive<scalar_t><<<dim3(max_seq,q.size(1),cu.numel()-1),128,0,stream>>>(
      (const scalar_t*)q.data_ptr(),(const scalar_t*)k.data_ptr(),(const scalar_t*)v.data_ptr(),
      (scalar_t*)out.data_ptr(),cu.data_ptr<int32_t>(),s3(q),s3(k),s3(v),q.size(1),q.size(2),scale);
    else if(q.size(2)<=64) attention_select<scalar_t,64>(q,k,v,out,cu,max_seq,scale,impl,stream);
    else if(q.size(2)<=80) attention_select<scalar_t,80>(q,k,v,out,cu,max_seq,scale,impl,stream);
    else if(q.size(2)<=128) attention_select<scalar_t,128>(q,k,v,out,cu,max_seq,scale,impl,stream);
    else attention_select<scalar_t,256>(q,k,v,out,cu,max_seq,scale,impl,stream);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

__device__ float decay_value(float alog,float a,float bias,float beta,float threshold) {
  float x=a+bias;
  float sp=beta*x<=threshold ? log1pf(expf(beta*x))/beta : x;
  return expf(-expf(alog)*sp);
}
template<class T>
__global__ void gates_kernel(const float* alog,const T* a,const T* b,const float* bias,
    float* gates,int64_t count,int hv,float beta,float threshold) {
  int64_t z=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(z>=count) return;
  gates[z*2]=decay_value(alog[z%hv],as_float(a[z]),bias[z%hv],beta,threshold);
  gates[z*2+1]=1.f/(1.f+expf(-as_float(b[z])));
}
template<class T>
__global__ void normalize_kernel(const T* q,const T* k,float* qn,float* kn,
    int dim,bool norm,float scale) {
  int lane=threadIdx.x; int64_t base=int64_t(blockIdx.x)*dim;
  float qsum=0,ksum=0;
  for(int d=lane;d<dim;d+=32) {
    float x=as_float(q[base+d]),y=as_float(k[base+d]);
    qsum+=x*x;ksum+=y*y;
  }
  qsum=warp_sum(qsum);ksum=warp_sum(ksum);
  float qi=norm ? rsqrtf(qsum+1e-6f) : 1.f,ki=norm ? rsqrtf(ksum+1e-6f) : 1.f;
  for(int d=lane;d<dim;d+=32) {
    qn[base+d]=as_float(q[base+d])*qi*scale;
    kn[base+d]=as_float(k[base+d])*ki;
  }
}
template<class T,class State,bool Fused>
__global__ void delta_kernel(const T* q,const T* k,const T* v,const float* alog,
    const T* a,const T* b,const float* bias,State* state,State* trace,T* out,
    const int32_t* cu,const int32_t* indices,const int32_t* accepted,
    const float* gates,const float* qn,const float* kn,int heads,int hv,int dk,int dv,
    int states,int iw,float scale,float beta,float threshold,bool norm,bool inplace) {
  int lane=threadIdx.x%32,vr=blockIdx.x*4+threadIdx.x/32,h=blockIdx.y,n=blockIdx.z;
  if(vr>=dv) return;
  int beg=cu[n],end=cu[n+1];
  if(beg==end) return;
  int it=accepted ? accepted[n]-1 : 0;
  int init=(it>=0 && it<iw) ? indices[n*iw+it] : 0;
  if(init<=0 || init>=states) {
    if(lane==0) for(int t=beg;t<end;++t) out[(int64_t(t)*hv+h)*dv+vr]=from_float<T>(0);
    if(!inplace) for(int t=beg;t<end;++t) for(int d=lane;d<dk;d+=32)
      trace[((int64_t(t)*hv+h)*dv+vr)*dk+d]=from_float<State>(0);
    return;
  }
  int kh=h/(hv/heads);
  float s[8];
  #pragma unroll
  for(int u=0;u<8;++u) {
    int d=lane+u*32;
    s[u]=d<dk ? as_float(state[((int64_t(init)*hv+h)*dv+vr)*dk+d]) : 0;
  }
  for(int t=beg;t<end;++t) {
    float qq[8],kk[8],qsum=0,ksum=0;
    #pragma unroll
    for(int u=0;u<8;++u) {
      int d=lane+u*32;int64_t z=(int64_t(t)*heads+kh)*dk+d;
      if constexpr(Fused) {
        qq[u]=d<dk ? as_float(q[z]) : 0;
        kk[u]=d<dk ? as_float(k[z]) : 0;
        qsum+=qq[u]*qq[u];ksum+=kk[u]*kk[u];
      } else {qq[u]=d<dk ? qn[z] : 0;kk[u]=d<dk ? kn[z] : 0;}
    }
    if constexpr(Fused) {
      qsum=warp_sum(qsum);ksum=warp_sum(ksum);
      float qi=(norm ? rsqrtf(qsum+1e-6f) : 1.f)*scale;
      float ki=norm ? rsqrtf(ksum+1e-6f) : 1.f;
      #pragma unroll
      for(int u=0;u<8;++u) {qq[u]*=qi;kk[u]*=ki;}
    }
    float decay,beta_t;
    if constexpr(Fused) {
      decay=decay_value(alog[h],as_float(a[t*hv+h]),bias[h],beta,threshold);
      beta_t=1.f/(1.f+expf(-as_float(b[t*hv+h])));
    } else {decay=gates[(t*hv+h)*2];beta_t=gates[(t*hv+h)*2+1];}
    float pred=0;
    #pragma unroll
    for(int u=0;u<8;++u) {s[u]*=decay;pred+=s[u]*kk[u];}
    pred=warp_sum(pred);
    float delta=(as_float(v[(int64_t(t)*hv+h)*dv+vr])-pred)*beta_t, o=0;
    #pragma unroll
    for(int u=0;u<8;++u) {s[u]+=delta*kk[u];o+=s[u]*qq[u];}
    o=warp_sum(o);
    if(lane==0) out[(int64_t(t)*hv+h)*dv+vr]=from_float<T>(o);
    int dest=inplace ? (t-beg<iw ? indices[n*iw+t-beg] : 0) : t;
    if(!inplace || (dest>0 && dest<states)) {
      State* ptr=inplace ? state : trace;
      #pragma unroll
      for(int u=0;u<8;++u) {
        int d=lane+u*32;
        if(d<dk) ptr[((int64_t(dest)*hv+h)*dv+vr)*dk+d]=from_float<State>(s[u]);
      }
    }
  }
}
// Share normalization and gates across all 16 V rows in a CTA. Keep each
// row's state in registers; all threads, including padded V rows, join barriers.
template<class T,class State,int Slots,int Rows>
__global__ void delta_shared_kernel(const T* q,const T* k,const T* v,const float* alog,
    const T* a,const T* b,const float* bias,State* state,State* trace,T* out,
    const int32_t* cu,const int32_t* indices,const int32_t* accepted,
    const float*,const float*,const float*,int heads,int hv,int dk,int dv,
    int states,int iw,float scale,float beta,float threshold,bool norm,bool inplace) {
  int lane=threadIdx.x%32,warp=threadIdx.x/32,h=blockIdx.y,n=blockIdx.z;
  int beg=cu[n],end=cu[n+1];
  if(beg==end) return;
  int it=accepted ? accepted[n]-1 : 0;
  int init=(it>=0 && it<iw) ? indices[n*iw+it] : 0;
  if(init<=0 || init>=states) {
    for(int r=0;r<Rows;++r) {
      int vr=blockIdx.x*(4*Rows)+warp*Rows+r;
      if(vr>=dv) continue;
      if(lane==0) for(int t=beg;t<end;++t) out[(int64_t(t)*hv+h)*dv+vr]=from_float<T>(0);
      if(!inplace) for(int t=beg;t<end;++t) for(int d=lane;d<dk;d+=32)
        trace[((int64_t(t)*hv+h)*dv+vr)*dk+d]=from_float<State>(0);
    }
    return;
  }
  __shared__ float qbuf[Slots*32],kbuf[Slots*32],gates[2];
  float state_rows[Rows][Slots];
  #pragma unroll
  for(int r=0;r<Rows;++r) {
    int vr=blockIdx.x*(4*Rows)+warp*Rows+r;
    #pragma unroll
    for(int u=0;u<Slots;++u) {
      int d=lane+u*32;
      state_rows[r][u]=(vr<dv && d<dk) ? as_float(state[((int64_t(init)*hv+h)*dv+vr)*dk+d]) : 0;
    }
  }
  int kh=h/(hv/heads);
  for(int t=beg;t<end;++t) {
    if(warp==0) {
      float qq[Slots],kk[Slots],qsum=0,ksum=0;
      #pragma unroll
      for(int u=0;u<Slots;++u) {
        int d=lane+u*32;int64_t z=(int64_t(t)*heads+kh)*dk+d;
        qq[u]=d<dk ? as_float(q[z]) : 0;kk[u]=d<dk ? as_float(k[z]) : 0;
        qsum+=qq[u]*qq[u];ksum+=kk[u]*kk[u];
      }
      qsum=warp_sum(qsum);ksum=warp_sum(ksum);
      float qi=norm ? rsqrtf(qsum+1e-6f) : 1.f,ki=norm ? rsqrtf(ksum+1e-6f) : 1.f;
      #pragma unroll
      for(int u=0;u<Slots;++u) {qbuf[lane+u*32]=qq[u]*qi*scale;kbuf[lane+u*32]=kk[u]*ki;}
      if(lane==0) {
        gates[0]=decay_value(alog[h],as_float(a[t*hv+h]),bias[h],beta,threshold);
        gates[1]=1.f/(1.f+expf(-as_float(b[t*hv+h])));
      }
    }
    __syncthreads();
    float qq[Slots],kk[Slots];
    #pragma unroll
    for(int u=0;u<Slots;++u) {qq[u]=qbuf[lane+u*32];kk[u]=kbuf[lane+u*32];}
    #pragma unroll
    for(int r=0;r<Rows;++r) {
      int vr=blockIdx.x*(4*Rows)+warp*Rows+r;
      float pred=0;
      #pragma unroll
      for(int u=0;u<Slots;++u) {state_rows[r][u]*=gates[0];pred+=state_rows[r][u]*kk[u];}
      pred=warp_sum(pred);
      float residual=vr<dv ? (as_float(v[(int64_t(t)*hv+h)*dv+vr])-pred)*gates[1] : 0;
      float value=0;
      #pragma unroll
      for(int u=0;u<Slots;++u) {state_rows[r][u]+=residual*kk[u];value+=state_rows[r][u]*qq[u];}
      value=warp_sum(value);
      if(lane==0 && vr<dv) out[(int64_t(t)*hv+h)*dv+vr]=from_float<T>(value);
      int dest=inplace ? (t-beg<iw ? indices[n*iw+t-beg] : 0) : t;
      if(vr<dv && (!inplace || (dest>0 && dest<states))) {
        State* ptr=inplace ? state : trace;
        #pragma unroll
        for(int u=0;u<Slots;++u) {
          int d=lane+u*32;
          if(d<dk) ptr[((int64_t(dest)*hv+h)*dv+vr)*dk+d]=from_float<State>(state_rows[r][u]);
        }
      }
    }
    if(t+1<end) __syncthreads();
  }
}

std::vector<at::Tensor> delta_cuda(at::Tensor q,at::Tensor k,at::Tensor v,
    at::Tensor alog,at::Tensor a,at::Tensor b,at::Tensor bias,at::Tensor state,
    at::Tensor cu,at::Tensor indices,at::Tensor accepted,double scale,double beta,
    double threshold,bool norm,bool inplace,int64_t impl) {
  c10::cuda::CUDAGuard guard(q.device());
  auto stream=at::cuda::getCurrentCUDAStream(q.get_device());
  int t=q.size(0),h=q.size(1),dk=q.size(2),hv=v.size(1),dv=v.size(2);
  bool fused=impl!=0;
  auto out=at::empty(v.sizes(),v.options());
  auto trace=inplace ? state : at::empty({t,hv,dv,dk},state.options());
  if(!t) return {out,trace};
  at::Tensor gates,qn,kn;
  if(!fused) {
    gates=at::empty({t,hv,2},q.options().dtype(at::kFloat));
    qn=at::empty(q.sizes(),q.options().dtype(at::kFloat));
    kn=at::empty(k.sizes(),k.options().dtype(at::kFloat));
  }
  #define DELTA_ARGS (const scalar_t*)q.data_ptr(),(const scalar_t*)k.data_ptr(), \
    (const scalar_t*)v.data_ptr(),alog.data_ptr<float>(),(const scalar_t*)a.data_ptr(), \
    (const scalar_t*)b.data_ptr(),bias.data_ptr<float>(),(state_t*)state.data_ptr(), \
    (state_t*)trace.data_ptr(),(scalar_t*)out.data_ptr(),cu.data_ptr<int32_t>(), \
    indices.data_ptr<int32_t>(),accepted.numel() ? accepted.data_ptr<int32_t>() : nullptr, \
    fused ? nullptr : gates.data_ptr<float>(),fused ? nullptr : qn.data_ptr<float>(), \
    fused ? nullptr : kn.data_ptr<float>(),h,hv,dk,dv,int(state.size(0)),int(indices.size(1)), \
    float(scale),float(beta),float(threshold),norm,inplace
  INPUT_DISPATCH(q, {
    if(!fused) {
      gates_kernel<scalar_t><<<(int64_t(t)*hv+127)/128,128,0,stream>>>(
        alog.data_ptr<float>(),(const scalar_t*)a.data_ptr(),(const scalar_t*)b.data_ptr(),
        bias.data_ptr<float>(),gates.data_ptr<float>(),int64_t(t)*hv,hv,beta,threshold);
      normalize_kernel<scalar_t><<<t*h,32,0,stream>>>((const scalar_t*)q.data_ptr(),
        (const scalar_t*)k.data_ptr(),qn.data_ptr<float>(),kn.data_ptr<float>(),dk,norm,scale);
    }
    STATE_DISPATCH(state, {
      if(impl==2) {
        delta_shared_kernel<scalar_t,state_t,8,1><<<dim3((dv+3)/4,hv,cu.numel()-1),128,0,stream>>>(DELTA_ARGS);
      }
      else if(impl==3) {
        delta_shared_kernel<scalar_t,state_t,8,4><<<dim3((dv+15)/16,hv,cu.numel()-1),128,0,stream>>>(DELTA_ARGS);
      }
      else if(impl==4) {
        if(dk<=128) delta_shared_kernel<scalar_t,state_t,4,4><<<dim3((dv+15)/16,hv,cu.numel()-1),128,0,stream>>>(DELTA_ARGS);
        else delta_shared_kernel<scalar_t,state_t,8,4><<<dim3((dv+15)/16,hv,cu.numel()-1),128,0,stream>>>(DELTA_ARGS);
      }
      else if(fused) delta_kernel<scalar_t,state_t,true><<<dim3((dv+3)/4,hv,cu.numel()-1),128,0,stream>>>(DELTA_ARGS);
      else delta_kernel<scalar_t,state_t,false><<<dim3((dv+3)/4,hv,cu.numel()-1),128,0,stream>>>(DELTA_ARGS);
    });
  });
  #undef DELTA_ARGS
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out,trace};
}
