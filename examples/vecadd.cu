#include	<stdio.h>
#include	<float.h>
#include	"cuda_auxiliary.h"


__global__	void	gpu_vec_add(double *a, double *b, double *c, int N)
{
	const int ix = blockIdx.x * blockDim.x + threadIdx.x;

	if (ix < N)
		c[ix] = a[ix] + b[ix];
}

void cpu_vec_add(double *a, double *b, double *c, int N)
{
	for (int i = 0; i < N; ++i)
		c[i] = a[i] + b[i];
}

void check_result(double *hst_c, double *dev_c, int N)
{
	for (int i = 0; i < N; ++i)
		if (abs(hst_c[i] - dev_c[i]) >= DBL_EPSILON) {
			printf("CPU and GPU results differ at position %d\n", i);
			return;
		}

	printf("GPU result is correct\n");
}


int main(int argc, char **argv)
{
	int				N = 1 << 20;

	double			*hst_a = NULL;
	double			*hst_b = NULL;
	double			*hst_c = NULL;
	double			*hst_r = NULL;

	double			*dev_a = NULL;
	double			*dev_b = NULL;
	double			*dev_c = NULL;

	dim3			block;
	dim3			grid;

	double			cpu_time = 0.0;
	double			gpu_time = 0.0;


	if (argc != 2) {
		fprintf(stderr, "usage: %s dimx\n", argv[0]);
		goto die;
	}

	host_alloc(hst_a, double, N * sizeof(double));
	host_alloc(hst_b, double, N * sizeof(double));
	host_alloc(hst_c, double, N * sizeof(double));
	host_alloc(hst_r, double, N * sizeof(double));

	init_matrix(hst_a, N, 1, N);
	init_matrix(hst_b, N, 1, N);	

	cuda_exec(cudaMalloc(&dev_a, N * sizeof(double)));
	cuda_exec(cudaMalloc(&dev_b, N * sizeof(double)));
	cuda_exec(cudaMalloc(&dev_c, N * sizeof(double)));

	cuda_exec(cudaMemcpy(dev_a, hst_a, N * sizeof(double), cudaMemcpyHostToDevice));
	cuda_exec(cudaMemcpy(dev_b, hst_b, N * sizeof(double), cudaMemcpyHostToDevice));

	block.x = atoi(argv[1]);
	grid.x = (N + block.x - 1) / block.x;

	cpu_time -= timer();
	cpu_vec_add(hst_a, hst_b, hst_c, N);
	cpu_time += timer();

	gpu_time -= timer();
	gpu_vec_add<<<grid, block>>>(dev_a, dev_b, dev_c, N);	
	cuda_exec(cudaDeviceSynchronize());
	gpu_time += timer();

	cuda_exec(cudaMemcpy(hst_r, dev_c, N * sizeof(double), cudaMemcpyDeviceToHost));
	cuda_exec(cudaDeviceSynchronize());

	printf("CPU time: %#.6lgs\n", cpu_time);
	printf("GPU time: %#.6lgs\n", gpu_time);

	check_result(hst_c, hst_r, N);
	
	
die:
	free(hst_a);
	free(hst_b);
	free(hst_c);

	cuda_exec(cudaFree(dev_a));
	cuda_exec(cudaFree(dev_b));
	cuda_exec(cudaFree(dev_c));

	return 0;
}
