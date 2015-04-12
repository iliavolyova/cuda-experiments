#include	<stdio.h>
#include	<float.h>
#include	"cuda_auxiliary.h"
#include "CImg.h"
using namespace cimg_library;

__global__ void gpu_mv_mul(double *vec, double *mat, double *out, const int N, const int M){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    double sum = 0;
    if(tid < M){
        for(int i = 0; i < N; i++)
            sum += vec[i] * mat[(i*M) + tid];
        out[tid] = sum;
    }
}

int main(int argc, char **argv) {
    double *hst_A = NULL;
    double *hst_B = NULL;
    double *hst_C = NULL;

    double *dev_A = NULL;
    double *dev_B = NULL;
    double *dev_C = NULL;

    dim3 block_size;
    dim3 grid_size;

    if (argc != 3) {
        fprintf(stderr, "usage: %s dimx dimy\n", argv[0]);
        return 0;
    }

    CImg<double> src("neno.png");
    int nx = src.width();
    int ny = src.height();
    int depth = src.depth();

    hst_A = src.data();
    double sepia[9] = {
        0.393, 0.769, 0.189,
        0.349, 0.686, 0.168,
        0.272, 0.534, 0.131
    };
    hst_B = sepia;

    host_alloc(hst_C, double, nx * ny * 3 * sizeof(double));

    cuda_exec(cudaMalloc(&dev_A, nx * ny * 3 * sizeof(double)));
    cuda_exec(cudaMalloc(&dev_B, 3 * 3 * sizeof(double)));
    cuda_exec(cudaMalloc(&dev_C, nx * ny * 3 * sizeof(double)));

    cuda_exec(cudaMemcpy(dev_A, hst_A, nx * ny * sizeof(double), cudaMemcpyHostToDevice));
    cuda_exec(cudaMemcpy(dev_B, hst_B, nx * ny * sizeof(double), cudaMemcpyHostToDevice));

    block_size.x = atoi(argv[1]);
    block_size.y = atoi(argv[2]);

    grid_size.x = min((nx + block_size.x - 1) / block_size.x, 65535);
    grid_size.y = min((ny + block_size.y - 1) / block_size.y, 65535);

    gpu_mv_mul<<<grid_size, block_size>>>(dev_A, dev_B, dev_C, 3, ny);
    cuda_exec(cudaDeviceSynchronize());

    cuda_exec(cudaMemcpy(hst_C, dev_C, nx * ny * 3 * sizeof(double), cudaMemcpyDeviceToHost));

    CImg<double> res(hst_A,nx,ny,depth,3,false);
    res.save("neno.bmp");

    free(hst_A);
    free(hst_B);
    free(hst_C);

    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C);

    return 0;
}