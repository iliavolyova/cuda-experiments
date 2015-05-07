#include	<stdio.h>
#include	<float.h>
#include	"cuda_auxiliary.h"

typedef struct {
	double *x;
	double *y;
	double *z;
} points;


__global__	void	gpu_norm(points array, double *norm, int N)
{
	for (int ix = blockIdx.x * blockDim.x + threadIdx.x; ix < N; ix += blockDim.x * gridDim.x)
		norm[ix] = sqrt(array.x[ix] * array.x[ix] + array.y[ix] * array.y[ix] + array.z[ix] * array.z[ix]);
}

void cpu_norm(points array, double *norm, int N)
{
	for (int ix = 0; ix < N; ++ix)
		norm[ix] = sqrt(array.x[ix] * array.x[ix] + array.y[ix] * array.y[ix] + array.z[ix] * array.z[ix]);
}

void check_result(double *cpu_c, double *gpu_c, int N)
{
	for (int i = 0; i < N; ++i)
		if (abs(cpu_c[i] - gpu_c[i]) >= 3 * DBL_EPSILON) {
			printf("CPU and GPU results differ at position %d\n", i);
			printf("CPU value: %lg\n", cpu_c[i]);
			printf("GPU value: %lg\n", gpu_c[i]);
			return;
		}

	printf("GPU result is correct\n");
}


int main(int argc, char **argv)
{
	int				N = 1 << 24;

	points			hst_point;
	points			dev_point;

	double			*hst_n = NULL;
	double			*hst_r = NULL;
	double			*dev_n = NULL;

	dim3			block;
	dim3			grid;

	double			cpu_time = 0.0;
	double			gpu_time = 0.0;


	if (argc != 2) {
		fprintf(stderr, "usage: %s dimx\n", argv[0]);
		goto die;
	}

	host_alloc(hst_point.x, double, 3 * N);
	host_alloc(hst_n, double, N);
	host_alloc(hst_r, double, N);

	cuda_exec(cudaMalloc(&dev_point.x, 3 * N * sizeof(double)));
	cuda_exec(cudaMalloc(&dev_n, N * sizeof(double)));

	hst_point.y = hst_point.x + N;
	hst_point.z = hst_point.y + N;

	dev_point.y = dev_point.x + N;
	dev_point.z = dev_point.y + N;

	init_matrix(hst_point.x, 3 * N, 1, 3 * N);

	cuda_exec(cudaMemcpy(dev_point.x, hst_point.x, 3 * N * sizeof(double), cudaMemcpyHostToDevice));

	block.x = atoi(argv[1]);
	grid.x = min((N + block.x - 1) / block.x, 65535);

	cpu_time -= timer();
	cpu_norm(hst_point, hst_n, N);
	cpu_time += timer();

	gpu_time -= timer();
	gpu_norm<<<grid, block>>>(dev_point, dev_n, N);	
	cuda_exec(cudaDeviceSynchronize());
	gpu_time += timer();

	cuda_exec(cudaMemcpy(hst_r, dev_n, N * sizeof(double), cudaMemcpyDeviceToHost));
	cuda_exec(cudaDeviceSynchronize());

	printf("Execution configuration: %d blocks, %d threads\n", grid.x, block.x);
	printf("CPU time: %dms\n", (int) (1000 * cpu_time));
	printf("GPU time: %dms\n", (int) (1000 * gpu_time));

	check_result(hst_n, hst_r, N);
	
	
die:
	free(hst_point.x);
	free(hst_n);
	free(hst_r);

	cuda_exec(cudaFree(dev_point.x));
	cuda_exec(cudaFree(dev_n));

	return 0;
}
