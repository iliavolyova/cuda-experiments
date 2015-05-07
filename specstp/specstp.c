#include	<stdio.h>
#include    "cuda_auxiliary.h"

/*
 * compile: nvcc specstp.cu -o specstp
 * trebaju ti dgemv, dnrm2
 */


int main(int argc, char **argv)
{
    cublas_handle_t cublas_handle;
    cublas_exec(cublasCreate(&cublas_handle));

    int version;
    cublas_exec(cublasGetVersion(cublas_handle, &version));
    printf("\nUsing CuBlas version: %d", version);

    cublas_exec(cublasDestroy(cublas_handle));

    return 0;
}

