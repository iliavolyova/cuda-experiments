#include	<stdio.h>
#include	<cublas_v2.h>
#include    "cuda_auxiliary.h"

/*
 * compile: nvcc specstp.cu -lcublas -o specstp
 * trebaju ti dgemv, dnrm2
 */


int main(int argc, char **argv)
{
    cublasHandle_t cublas_handle;

    FILE *fp_A = NULL;
    FILE *fp_x = NULL;

    double *hst_A = NULL;
    double *hst_x = NULL;

    double *dev_A = NULL;
    double *dev_x = NULL;

    double ONE = 1.0;

    int dim;
    int steps;

    if (argc != 5) {
        fprintf(stderr, "usage: %s N A.dat x0.dat steps\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    dim = atoi(argv[1]);
    steps = atoi(argv[4]);

    open_file(fp_A, argv[2], "r");
    host_alloc(hst_A, double, dim * dim);

    open_file(fp_x, argv[3], "r");
    host_alloc(hst_x, double, dim);

    cuda_exec(cudaHostRegister(hst_A, dim * dim * sizeof(double), cudaHostRegisterDefault));
    read_file(hst_A, sizeof(double), dim * dim, fp_A);

    cuda_exec(cudaHostRegister(hst_x, dim * sizeof(double), cudaHostRegisterDefault));
    read_file(hst_x, sizeof(double), dim, fp_x);

    cuda_exec(cudaMalloc(&dev_A, dim * dim * sizeof(double)));
    cuda_exec(cudaMalloc(&dev_x, dim * sizeof(double)));

    cublas_exec(cublasCreate(&cublas_handle));
    cublas_exec(cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_DEVICE));

    cublas_exec(cublasSetMatrix(dim, dim, sizeof(double), hst_A, dim, dev_A, dim));
    cublas_exec(cublasSetVector(dim, sizeof(double), hst_x, ONE, dev_x, ONE));

    int i;
    for (i = 0; i < steps; i++){

    }

    printf("done");

    cublas_exec(cublasDestroy(cublas_handle));
    cudaFree(dev_A);
    cudaFree(dev_x);

    host_free(hst_A);
    host_free(hst_x);

    return 0;
}

