#include	<stdio.h>
#include	<float.h>
#include	"cuda_auxiliary.h"

__global__ void gpu_mat_add(double *A, double *B, double *C, const int nx, const int ny)
{
	unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;

	if (ix < nx)
		for (int iy = 0; iy < ny; ++iy) {
			int idx = iy * nx + ix;

			C[idx] = A[idx] + B[idx];
		}
}

void cpu_mat_add(double *A, double *B, double *C, const int nx, const int ny)
{
	for (int iy = 0; iy < ny; ++iy) {
		for (int ix = 0; ix < nx; ++ix)
			C[ix] = A[ix] + B[ix];

		A += nx;
		B += nx;
		C += nx;
	}
}

void check_results(double *cpu_array, double *gpu_array, int size)
{
	for (int ix = 0; ix < size; ++ix)
		if (abs(cpu_array[ix] - gpu_array[ix]) >= DBL_EPSILON) {
			printf("CPU and GPU results differ at element %d\n", ix);
			printf("CPU value: %lg\n", cpu_array[ix]);
			printf("GPU value: %lg\n", gpu_array[ix]);

			return;
		}

	printf("GPU result is correct\n");
}


int main(int argc, char **argv)
{
	double	*hst_A = NULL;
	double	*hst_B = NULL;
	double	*hst_C = NULL;

	double  *dev_A = NULL;
	double	*dev_B = NULL;
	double	*dev_C = NULL;

	int		nx = 1 << 13;
	int		ny = 1 << 13;

	double	cpu_time = 0.0;
	double	gpu_time = 0.0;

	dim3	block_size;
	dim3	grid_size;


	if (argc != 2) {
		fprintf(stderr, "usage: %s dimx\n", argv[0]);
		exit(EXIT_FAILURE);
	}


	host_alloc(hst_A, double, nx * ny);
	host_alloc(hst_B, double, nx * ny);
	host_alloc(hst_C, double, nx * ny);

	cuda_exec(cudaMalloc(&dev_A, nx * ny * sizeof(double)));
	cuda_exec(cudaMalloc(&dev_B, nx * ny * sizeof(double)));
	cuda_exec(cudaMalloc(&dev_C, nx * ny * sizeof(double)));


	init_matrix(hst_A, nx, ny, nx);
	init_matrix(hst_B, nx, ny, nx);

	cuda_exec(cudaMemcpy(dev_A, hst_A, nx * ny * sizeof(double), cudaMemcpyHostToDevice));
	cuda_exec(cudaMemcpy(dev_B, hst_B, nx * ny * sizeof(double), cudaMemcpyHostToDevice));


	block_size.x = atoi(argv[1]);
	block_size.y = 1;

	grid_size.x = min((nx + block_size.x - 1) / block_size.x, 65535);
	grid_size.y = 1;


	gpu_time -= timer();
	gpu_mat_add<<<grid_size, block_size>>>(dev_A, dev_B, dev_C, nx, ny);	
	cuda_exec(cudaDeviceSynchronize());
	gpu_time += timer();

	cpu_time -= timer();
	cpu_mat_add(hst_A, hst_B, hst_C, nx, ny);
	cpu_time += timer();
	
	cuda_exec(cudaMemcpy(hst_B, dev_C, nx * ny * sizeof(double), cudaMemcpyDeviceToHost));
	check_results(hst_C, hst_B, nx * ny);

	printf("Execution configuration: grid (%d, %d), block (%d, %d)\n", grid_size.x, grid_size.y, block_size.x, block_size.y);
	printf("CPU time: %.3lgms\n", 1000 * cpu_time);
	printf("GPU time: %.3lgms\n", 1000 * gpu_time);
	
	free(hst_A);
	free(hst_B);
	free(hst_C);

	cudaFree(dev_A);
	cudaFree(dev_B);
	cudaFree(dev_C);

	return 0;
}
	

