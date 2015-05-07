#include	<stdio.h>
#include	<float.h>
#include	"cuda_auxiliary.h"

typedef struct {
	double x;
	double y;
	double z;
} point;


__global__	void	gpu_norm(point *array, double *norm, int N)
{
	for (int ix = blockIdx.x * blockDim.x + threadIdx.x; ix < N; ix += blockDim.x * gridDim.x)
		norm[ix] = sqrt(array[ix].x * array[ix].x + array[ix].y * array[ix].y + array[ix].z * array[ix].z);
}

void cpu_norm(point *array, double *norm, int N)
{
	for (int ix = 0; ix < N; ++ix)
		norm[ix] = sqrt(array[ix].x * array[ix].x + array[ix].y * array[ix].y + array[ix].z * array[ix].z);
}

void check_result(double *cpu_c, double *gpu_c, int N)
{
	for (int i = 0; i < N; ++i)
		if (abs(cpu_c[i] - gpu_c[i]) >= 3 * DBL_EPSILON) {
			printf("CPU and GPU results differ at position %d\n", i);
			printf("CPU value: %#.16lg\n", cpu_c[i]);
			printf("GPU value: %#.16lg\n", gpu_c[i]);
			return;
		}

	printf("GPU result is correct\n");
}


int main(int argc, char **argv)
{
	int				N = 1 << 24;

	point			*hst_v = NULL;
	double			*hst_n = NULL;
	double			*hst_r = NULL;

	point			*dev_v = NULL;
	double			*dev_n = NULL;

	dim3			block;
	dim3			grid;

	double			cpu_time = 0.0;
	double			gpu_time = 0.0;


	if (argc != 2) {
		fprintf(stderr, "usage: %s dimx\n", argv[0]);
		exit(EXIT_FAILURE);
	}

	host_alloc(hst_v, point, N);
	host_alloc(hst_n, double, N);
	host_alloc(hst_r, double, N);

	init_matrix(((double *) hst_v) + 0, N, 1, 3);
	init_matrix(((double *) hst_v) + 1, N, 1, 3);
	init_matrix(((double *) hst_v) + 2, N, 1, 3);


	cuda_exec(cudaMalloc(&dev_v, N * sizeof(point)));
	cuda_exec(cudaMalloc(&dev_n, N * sizeof(double)));

	cuda_exec(cudaMemcpy(dev_v, hst_v, N * sizeof(point), cudaMemcpyHostToDevice));

	block.x = atoi(argv[1]);
	grid.x = min((N + block.x - 1) / block.x, 65535);

	cpu_time -= timer();
	cpu_norm(hst_v, hst_n, N);
	cpu_time += timer();

	gpu_time -= timer();
	gpu_norm<<<grid, block>>>(dev_v, dev_n, N);	
	cuda_exec(cudaDeviceSynchronize());
	gpu_time += timer();

	cuda_exec(cudaMemcpy(hst_r, dev_n, N * sizeof(double), cudaMemcpyDeviceToHost));
	cuda_exec(cudaDeviceSynchronize());

	printf("Execution configuration: %d blocks, %d threads\n", grid.x, block.x);
	printf("CPU time: %dms\n", (int) (1000 * cpu_time));
	printf("GPU time: %dms\n", (int) (1000 * gpu_time));

	check_result(hst_n, hst_r, N);
	
	
	free(hst_v);
	free(hst_n);
	free(hst_r);

	cuda_exec(cudaFree(dev_v));
	cuda_exec(cudaFree(dev_n));

	return 0;
}
