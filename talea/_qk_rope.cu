// Experiment for PyTorch 2.7.1's vectorized layer norm on an A5000.
// The Welford reduction order follows aten/native/cuda/layer_norm_kernel.cu.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

struct Stats { float mean, ss, n; };

__device__ __forceinline__ Stats add_one(Stats s, float x) {
    float n = s.n + 1.0f;
    float diff = x - s.mean;
    float m = s.mean + diff * (1.0f / n);
    return {m, s.ss + diff * (x - m), n};
}

__device__ __forceinline__ Stats merge(Stats b, Stats a) {
    float n = a.n + b.n;
    if (n == 0.0f) return {0.0f, 0.0f, 0.0f};
    float h = b.mean - a.mean;
    float inv = 1.0f / n;
    float na = a.n * inv, nb = b.n * inv;
    return {na * a.mean + nb * b.mean, a.ss + b.ss + h*h*a.n*nb, n};
}

struct alignas(8) Half4 {half h[4];};
constexpr int d = 1152;

__global__ void qk_rope(const half* packed, const half* qw, const half* kw,
                         const half* cs, const half* sn, half* q, half* k,
                         int tokens, float eps) {
    const int row = blockIdx.x;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const bool key = blockIdx.y != 0;
    const half* x = packed + (size_t(row)*3+key)*d;
    const half* gamma = key ? kw : qw;
    const Half4* in4 = reinterpret_cast<const Half4*>(x);
    Stats s{0.0f,0.0f,0.0f};
    for(int j=threadIdx.x; j<d/4; j+=blockDim.x) {
        Half4 v = in4[j];
        #pragma unroll
        for(int i=0; i<4; i++) s=add_one(s,__half2float(v.h[i]));
    }
    #pragma unroll
    for(int off=16; off>0; off>>=1) {
        Stats upper{__shfl_down_sync(0xffffffff,s.mean,off),
                    __shfl_down_sync(0xffffffff,s.ss,off),
                    __shfl_down_sync(0xffffffff,s.n,off)};
        s=merge(s,upper);
    }
    __shared__ Stats ws[4];
    if(!lane) ws[warp]=s;
    __syncthreads();
    if(!threadIdx.x) {
        Stats ab=merge(ws[0],ws[2]);
        Stats cd=merge(ws[1],ws[3]);
        Stats all=merge(ab,cd);
        ws[0]={all.mean,rsqrtf(all.ss/float(d)+eps),all.n};
    }
    __syncthreads();
    __shared__ half nrm[d];
    for(int j=threadIdx.x; j<d/4; j+=blockDim.x) {
        Half4 v=in4[j],w=reinterpret_cast<const Half4*>(gamma)[j], normalized;
        #pragma unroll
        for(int i=0; i<4; i++) normalized.h[i]=__float2half_rn(__half2float(w.h[i]) * (ws[0].ss * (__half2float(v.h[i])-ws[0].mean)));
        reinterpret_cast<Half4*>(nrm)[j]=normalized;
    }
    __syncthreads();
    Half4* out=reinterpret_cast<Half4*>((key?k:q)+size_t(row)*d);
    int table=(row%tokens)*32;
    for(int j=threadIdx.x;j<d/4;j+=blockDim.x) {
        Half4 here=reinterpret_cast<Half4*>(nrm)[j], peer=reinterpret_cast<Half4*>(nrm)[j^8], value;
        Half4 c=reinterpret_cast<const Half4*>(cs+table)[j%8], t=reinterpret_cast<const Half4*>(sn+table)[j%8];
        #pragma unroll
        for(int i=0;i<4;i++) {
            half y=((j&8)==0)?__hneg(peer.h[i]):peer.h[i];
            value.h[i]=__hadd_rn(__hmul_rn(here.h[i],c.h[i]),__hmul_rn(y,t.h[i]));
        }
        out[j]=value;
    }
}

extern "C" int run_qk(const half* packed, const half* qw, const half* kw,
                      const half* cs, const half* sn, half* q, half* k,
                      int rows, int tokens, float eps, cudaStream_t stream) {
    qk_rope<<<dim3(rows,2),128,0,stream>>>(packed,qw,kw,cs,sn,q,k,tokens,eps);
    return int(cudaGetLastError());
}
