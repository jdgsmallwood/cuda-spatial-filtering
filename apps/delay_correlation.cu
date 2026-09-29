// Optional cuFFT backend for analyze_delay_samples.py. No instrument access.
#include <cuda_runtime.h>
#include <cufft.h>
#include <cstring>
#include <stdexcept>
#include <string>

namespace {
void check(cudaError_t status) {
  if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));
}
void check_fft(cufftResult status) {
  if(status!=CUFFT_SUCCESS)throw std::runtime_error("cuFFT error "+std::to_string(int(status)));
}
struct Buffer {
  void *p=nullptr;
  explicit Buffer(size_t bytes){check(cudaMalloc(&p,bytes));}
  ~Buffer(){if(p)cudaFree(p);}
  cufftComplex *c(){return static_cast<cufftComplex*>(p);}
};
struct Plan {
  cufftHandle handle=0;
  Plan(int length,int batches){check_fft(cufftPlan1d(&handle,length,CUFFT_C2C,batches));}
  ~Plan(){if(handle)cufftDestroy(handle);}
};
__global__ void prepare(const cufftComplex *in,cufftComplex *out,int windows,int n,int receptors) {
  size_t idx=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  size_t total=size_t(windows)*receptors*2*n;if(idx>=total)return;
  int t=idx%(2*n);size_t batch=idx/(2*n);int ant=batch%receptors;int w=batch/receptors;
  out[idx]=t<n?in[(size_t(w)*n+t)*receptors+ant]:make_cuFloatComplex(0,0);
}
__global__ void cross_spectra(const cufftComplex *x,const cufftComplex *y,
    const unsigned char *groups,cufftComplex *out,int windows,int length,int nx,int ny) {
  size_t idx=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  size_t total=size_t(3)*nx*ny*length;if(idx>=total)return;
  int f=idx%length;size_t pair=idx/length;int b=pair%ny;int a=(pair/ny)%nx;int group=pair/(nx*ny);
  float real=0,imag=0;
  for(int w=0;w<windows;++w)if(groups[group*windows+w]) {
    auto u=x[(size_t(w)*nx+a)*length+f];auto v=y[(size_t(w)*ny+b)*length+f];
    real+=u.x*v.x+u.y*v.y;imag+=u.y*v.x-u.x*v.y;
  }
  out[idx]=make_cuFloatComplex(real,imag);
}
__global__ void select_lags(const cufftComplex *in,cufftComplex *out,int length,int nx,int ny,int radius) {
  size_t idx=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  int nl=2*radius+1;size_t total=size_t(3)*nl*nx*ny;if(idx>=total)return;
  int b=idx%ny;int a=(idx/ny)%nx;int l=(idx/(nx*ny))%nl;int g=idx/(size_t(nx)*ny*nl);
  int f=(l-radius+length)%length;
  auto value=in[((size_t(g)*nx+a)*ny+b)*length+f];
  out[idx]=make_cuFloatComplex(value.x/length,value.y/length);
}
}

extern "C" int spatial_delay_correlations(const void *x,const void *y,
    const unsigned char *groups,int windows,int n,int nx,int ny,int radius,
    void *output,char *error,size_t error_size) {
  try {
    if(windows<1 || n<1 || nx<1 || ny<1 || radius<1 || radius>=n)
      throw std::runtime_error("invalid delay correlation dimensions");
    size_t xsize=size_t(windows)*n*nx,ysize=size_t(windows)*n*ny;
    size_t count=size_t(3)*(2*radius+1)*nx*ny;
    Buffer raw_x(xsize*sizeof(cufftComplex)),raw_y(ysize*sizeof(cufftComplex));
    Buffer X(2*xsize*sizeof(cufftComplex)),Y(2*ysize*sizeof(cufftComplex));
    Buffer G(3*windows),cross(size_t(3)*nx*ny*2*n*sizeof(cufftComplex));
    Buffer result(count*sizeof(cufftComplex));
    check(cudaMemcpy(raw_x.p,x,xsize*sizeof(cufftComplex),cudaMemcpyHostToDevice));
    check(cudaMemcpy(raw_y.p,y,ysize*sizeof(cufftComplex),cudaMemcpyHostToDevice));
    check(cudaMemcpy(G.p,groups,3*windows,cudaMemcpyHostToDevice));
    prepare<<<(2*xsize+255)/256,256>>>(raw_x.c(),X.c(),windows,n,nx);
    prepare<<<(2*ysize+255)/256,256>>>(raw_y.c(),Y.c(),windows,n,ny);
    check(cudaGetLastError());
    Plan px(2*n,windows*nx),py(2*n,windows*ny),pc(2*n,3*nx*ny);
    check_fft(cufftExecC2C(px.handle,X.c(),X.c(),CUFFT_FORWARD));
    check_fft(cufftExecC2C(py.handle,Y.c(),Y.c(),CUFFT_FORWARD));
    const size_t cross_count=size_t(3)*nx*ny*2*n;
    cross_spectra<<<(cross_count+255)/256,256>>>(X.c(),Y.c(),static_cast<unsigned char*>(G.p),cross.c(),windows,2*n,nx,ny);
    check(cudaGetLastError());
    check_fft(cufftExecC2C(pc.handle,cross.c(),cross.c(),CUFFT_INVERSE));
    select_lags<<<(count+255)/256,256>>>(cross.c(),result.c(),2*n,nx,ny,radius);
    check(cudaGetLastError());
    check(cudaMemcpy(output,result.p,count*sizeof(cufftComplex),cudaMemcpyDeviceToHost));
    return 0;
  }catch(const std::exception &e){
    if(error && error_size){std::strncpy(error,e.what(),error_size-1);error[error_size-1]=0;}
    return 1;
  }
}
