#include	<stdio.h>
#include    "cuda_auxiliary.h"

/*
 * compile: nvcc specstp.cu -lcublas -o specstp
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
    double *dev_y = NULL;

    double norm;
    double eigval;

    const double ONE = 1.0;
    const double ZERO = 0.0;
    double alpha;

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

    read_file(hst_A, sizeof(double), dim * dim, fp_A);
    read_file(hst_x, sizeof(double), dim, fp_x);

    cuda_exec(cudaMalloc(&dev_A, dim * dim * sizeof(double)));
    cuda_exec(cudaMalloc(&dev_x, dim * sizeof(double)));
    cuda_exec(cudaMalloc(&dev_y, dim * sizeof(double)));

    cublas_exec(cublasCreate(&cublas_handle));
    cublas_exec(cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_HOST));

    cublas_exec(cublasSetMatrix(dim, dim, sizeof(double), hst_A, dim, dev_A, dim));
    cublas_exec(cublasSetVector(dim, sizeof(double), hst_x, 1, dev_x, 1));
    cublas_exec(cublasSetVector(dim, sizeof(double), hst_x, 1, dev_y, 1));

    int i;
    for (i = 0; i < steps; ++i){
        cublas_exec(cublasDnrm2(cublas_handle, dim, dev_y, 1, &norm));
        alpha = 1.0/norm;
        cublas_exec(cublasDscal(cublas_handle, dim, &alpha, dev_y, 1));
        cublas_exec(cublasDcopy(cublas_handle, dim, dev_y, 1, dev_x, 1));

        cublas_exec(cublasDgemv(cublas_handle, CUBLAS_OP_T, dim, dim, &ONE, dev_A, dim, dev_x, 1, &ZERO, dev_y, 1));
    }

    cublas_exec(cublasDdot(cublas_handle, dim, dev_x, 1, dev_y, 1, &eigval));
    printf("\nSpectrum: %#.16lg\n", eigval);

    cublas_exec(cublasDestroy(cublas_handle));
    cudaFree(dev_A);
    cudaFree(dev_x);
    cudaFree(dev_y);

    host_free(hst_A);
    host_free(hst_x);

    return 0;
}

