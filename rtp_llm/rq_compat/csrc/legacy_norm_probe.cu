#include <cuda_runtime.h>
#include <cuda_fp16.h>
__global__ void legacy_norm_kernel(half* out,const half* in,const half* weight,int n,float eps){
 int t=threadIdx.x,r=blockIdx.x,l=t&31,w=t>>5; int h=n/2;
 extern __shared__ half2 cache[];
 __shared__ float partial[33],inv;
 float sum=0;
 for(int i=t;i<h;i+=blockDim.x){half2 a=((const half2*)in)[r*h+i];cache[i]=a;float2 v=__half22float2(a);sum+=v.x*v.x+v.y*v.y;}
 for(int m=16;m>0;m>>=1)sum+=__shfl_xor_sync(0xffffffff,sum,m,32);
 if(l==0)partial[w]=sum;
 __syncthreads();sum=t<blockDim.x/32?partial[l]:0;
 for(int m=16;m>0;m>>=1)sum+=__shfl_xor_sync(0xffffffff,sum,m,32);
 if(t==0)inv=rsqrtf(sum/n+eps);__syncthreads();
 for(int i=t;i<h;i+=blockDim.x){float2 a=__half22float2(cache[i]),g=__half22float2(((const half2*)weight)[i]);((half2*)out)[r*h+i]=__floats2half2_rn(a.x*inv*g.x,a.y*inv*g.y);}
}
extern "C" int rq_legacy_norm(half* out,const half* in,const half* weight,int rows,int n,float eps,cudaStream_t stream){
 if(n%2)return 1;int threads=n/2<1024?n/2:1024;threads=(threads+31)/32*32;
 legacy_norm_kernel<<<rows,threads,n*sizeof(half),stream>>>(out,in,weight,n,eps);return (int)cudaGetLastError();
}
