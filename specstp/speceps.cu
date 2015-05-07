#include	<stdio.h>
#include    "cuda_auxiliary.h"

#define BLOCK_SIZE 64

__global__ void gpu_dgemv(double *A, double *x, double *y, const int dim)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int i;
    double sum = 0;

    if (gid < dim){
        for (i = 0; i < dim; ++i){
            sum += A[(i*dim) + gid] * x[i];
        }
        y[gid] = sum;
    }

}

__global__ void gpu_dnrm2(double *x, double *nrm, const int dim)
{
    __shared__ double cache[BLOCK_SIZE];
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if (gid < dim)
        cache[tid] = x[gid];
    else
        cache[tid] = 0;

    __syncthreads();

    cache[tid] = cache[tid] * cache[tid];

    __syncthreads();

    int i = blockDim.x / 2;
    while(i > 0){
        if(tid < i)
            cache[tid] = cache[tid] + cache[tid + i];
        __syncthreads();

        i >>= 1;
    }

    if (tid == 0) {
        nrm[0] = sqrt(cache[0]);
    }
}

__global__ void gpu_dscal(double *x, double alpha, const int n)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    x[gid] *= alpha;
}

__global__ void gpu_ddot(double *x, double *y, double *out, const int dim)
{
    __shared__ double cache[BLOCK_SIZE];
    int cacheindex = threadIdx.x;
    double temp;

    for(int gid = blockIdx.x * blockDim.x + threadIdx.x; gid < dim; gid += blockDim.x * gridDim.x)
        temp += x[gid] * y[gid];

    cache[cacheindex] = temp;

    __syncthreads();

    int i = blockDim.x / 2;
    while (i > 0) {
        if (cacheindex < i)
            cache[cacheindex] += cache[cacheindex + i];
        __syncthreads();

        i >>= 1;
    }

    if (threadIdx.x = 0)
        out[0] = cache[0];

}

int main(int argc, char **argv)
{
    FILE *fp_A = NULL;
    FILE *fp_x = NULL;

    double *hst_A = NULL;
    double *hst_x = NULL;

    double *dev_A = NULL;
    double *dev_x = NULL;
    double *dev_y = NULL;
    double *dev_nrm = NULL;

    double norm;
    double eigval;

    const double ONE = 1.0;
    const double ZERO = 0.0;
    double alpha;
    bool converged = false;

    dim3	block_size;
    dim3	grid_size;

    int dim;

    if (argc != 4) {
        fprintf(stderr, "usage: %s N A.dat x0.dat\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    dim = atoi(argv[1]);

    open_file(fp_A, argv[2], "r");
    host_alloc(hst_A, double, dim * dim);

    open_file(fp_x, argv[3], "r");
    host_alloc(hst_x, double, dim);

    read_file(hst_A, sizeof(double), dim * dim, fp_A);
    read_file(hst_x, sizeof(double), dim, fp_x);

    cuda_exec(cudaMalloc(&dev_A, dim * dim * sizeof(double)));
    cuda_exec(cudaMalloc(&dev_x, dim * sizeof(double)));
    cuda_exec(cudaMalloc(&dev_y, dim * sizeof(double)));
    cuda_exec(cudaMalloc(&dev_nrm, sizeof(double)));

    cuda_exec(cudaMemcpy(dev_A, hst_A, dim * dim * sizeof(double), cudaMemcpyHostToDevice));
    cuda_exec(cudaMemcpy(dev_x, hst_x, dim * sizeof(double), cudaMemcpyHostToDevice));
    cuda_exec(cudaMemcpy(dev_y, hst_x, dim * sizeof(double), cudaMemcpyHostToDevice));

    block_size.x = BLOCK_SIZE;
    grid_size.x = min((dim + block_size.x - 1) / block_size.x, 65535);

    int i;
    while(!converged){
        gpu_dnrm2<<grid_size, block_size>>(dev_y, dev_nrm, dim);


        converged = true;
    }

    cuda_exec(cudaMemcpy(&eigval, dev_nrm, sizeof(double), cudaMemcpyDeviceToHost));


    printf("\nSpectrum: %#.16lg\n", eigval);

    cudaFree(dev_A);
    cudaFree(dev_x);
    cudaFree(dev_y);
    cudaFree(dev_nrm);

    host_free(hst_A);
    host_free(hst_x);

    return 0;
}

