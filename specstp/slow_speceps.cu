#include	<stdio.h>
#include    "cuda_auxiliary.h"

#define BLOCK_SIZE 64

__global__ void gpu_dgemv(double *A, double *x, double *y, const int dim)
{
    __shared__ double cache[BLOCK_SIZE];
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    double sum = 0;

    for (int i = 0; i < ((dim + BLOCK_SIZE - 1) / BLOCK_SIZE); ++i){
        if(i * BLOCK_SIZE + tid < dim)
            cache[tid] = x[threadIdx.x + i * BLOCK_SIZE];
        else
            cache[tid] = 0.f;
        __syncthreads();

        for (int j = 0; j < BLOCK_SIZE; ++j){
            sum += A[gid * dim + (i * BLOCK_SIZE + j)] * cache[j];
        }
        __syncthreads();
    }

    if(gid < dim)
        y[gid] = sum;

}

__global__ void gpu_dnrm2(double *x, double *nrm, const int dim, bool invert)
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


    for(int i = blockDim.x / 2; i > 0; i >>= 1){
        if(tid < i)
            cache[tid] = cache[tid] + cache[tid + i];
        __syncthreads();
    }

    if (tid == 0) {
        if (invert)
            nrm[0] = 1.0/sqrt(cache[0]);
        else
            nrm[0] = sqrt(cache[0]);
    }
}

__global__ void gpu_dscal(double *x, double *y, double *alpha, const int dim)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < dim)
        y[gid] = x[gid] * alpha[0];
}

__global__ void gpu_subtract(double *x, double *y, double *out, const int dim)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < dim)
        out[gid] = x[gid] - y[gid];
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

    for (int i = blockDim.x / 2; i > 0; i >>= 1) {
        if (cacheindex < i)
            cache[cacheindex] += cache[cacheindex + i];
        __syncthreads();
    }

    if (threadIdx.x == 0)
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
    double *dev_nrm_inv = NULL;
    double *dev_lambda;

    double eigval;
    double lambda;
    double subsnorm;
    double EPS = 0.00001;

    bool converged = false;

    dim3	block_size;
    dim3	grid_size;

    int dim;

    double dgemv_timer = 0.0;

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
    cuda_exec(cudaMalloc(&dev_nrm_inv, sizeof(double)));
    cuda_exec(cudaMalloc(&dev_lambda, sizeof(double)));

    cuda_exec(cudaMemcpy(dev_A, hst_A, dim * dim * sizeof(double), cudaMemcpyHostToDevice));
    cuda_exec(cudaMemcpy(dev_x, hst_x, dim * sizeof(double), cudaMemcpyHostToDevice));
    cuda_exec(cudaMemcpy(dev_y, hst_x, dim * sizeof(double), cudaMemcpyHostToDevice));

    block_size.x = BLOCK_SIZE;
    grid_size.x = min((dim + block_size.x - 1) / block_size.x, 65535);

    int cnt = 0;
    while(!converged){
        gpu_dnrm2<<<grid_size, block_size>>>(dev_y, dev_nrm_inv, dim, true);
        gpu_dscal<<<grid_size, block_size>>>(dev_y, dev_x, dev_nrm_inv, dim);

        if(cnt == 1)
            dgemv_timer -= timer();
        gpu_dgemv<<<grid_size, block_size>>>(dev_A, dev_x, dev_y, dim);
        if(cnt == 1)
            dgemv_timer += timer();

        gpu_ddot<<<grid_size, block_size>>>(dev_x, dev_y, dev_lambda, dim);

        gpu_dscal<<<grid_size, block_size>>>(dev_x, dev_x, dev_lambda, dim);
        gpu_subtract<<<grid_size, block_size>>>(dev_y, dev_x, dev_x, dim);
        gpu_dnrm2<<<grid_size, block_size>>>(dev_x, dev_nrm_inv, dim, false);

        cuda_exec(cudaMemcpy(&lambda, dev_lambda, sizeof(double), cudaMemcpyDeviceToHost));
        cuda_exec(cudaMemcpy(&subsnorm, dev_nrm_inv, sizeof(double), cudaMemcpyDeviceToHost));

        if (subsnorm < EPS * abs(lambda))
            converged = true;

        if (cnt == 100000){
            printf("died after %d iterations: %#.16lg > %#.16lg", cnt, subsnorm, EPS * lambda);
            break;
        }
        cnt++;
    }

    cuda_exec(cudaMemcpy(&eigval, dev_lambda, sizeof(double), cudaMemcpyDeviceToHost));
    printf("Dgemv took: %.3lgms\n", 1000* dgemv_timer);
    printf("Spectrum: %#.16lg, done after %d iterations\n", eigval, cnt);

    cudaFree(dev_A);
    cudaFree(dev_x);
    cudaFree(dev_y);
    cudaFree(dev_nrm_inv);
    cudaFree(dev_lambda);

    host_free(hst_A);
    host_free(hst_x);

    return 0;
}