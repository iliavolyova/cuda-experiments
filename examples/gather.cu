#include	<stdio.h>
#include	"cuda_auxiliary.h"

__global__ void gpu_gather(int *ix, int *data, int *a, int n)
{
	for (int idx = blockDim.x * blockIdx.x + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x)
		a[idx] = data[ix[idx]];
}

void cpu_gather(int *ix, int *data, int *a, int n)
{
	for (int i = 0; i < n; ++i)
		a[i] = data[ix[i]];
}

void check_result(int *hst_r, int *hst_a, int N)
{
	for (int i = 0; i < N; ++i)
		if (hst_r[i] != hst_a[i]) {
			printf("Arrays differ at element %d\n", i);
			printf("CPU value: %d\n", hst_a[i]);
			printf("GPU value: %d\n", hst_r[i]);
		}

	printf("GPU result is correct\n");
}

void init_indices(int *ix, int n, int offset)
{
	for (int i = 0; i < n; ++i)
		ix[i] = offset * i;
}

int main(int argc, char **argv)
{
	int		*hst_x = NULL;
	int		*hst_d = NULL;
	int		*hst_a = NULL;
	int		*hst_r = NULL;

	int		*dev_x = NULL;
	int		*dev_d = NULL;
	int		*dev_a = NULL;

	int		N = 10000000;
	int		M;

	dim3	grid;
	dim3	block;

	int		offset;

	double	cpu_time = 0.0;
	double	gpu_time = 0.0;

	if (argc != 3) {
		fprintf(stderr, "usage: %s dimx offset\n", argv[0]);
		exit(EXIT_FAILURE);
	}

	offset = atoi(argv[2]);
	M = offset * N;

	host_alloc(hst_x, int, N);
	host_alloc(hst_a, int, N);
	host_alloc(hst_r, int, N);
	host_alloc(hst_d, int, M);
	
	init_matrix(hst_d, M, 1, 1);
	init_indices(hst_x, N, offset);

	cuda_exec(cudaMalloc(&dev_x, N * sizeof(int)));
	cuda_exec(cudaMalloc(&dev_a, N * sizeof(int)));
	cuda_exec(cudaMalloc(&dev_d, M * sizeof(int)));

	cuda_exec(cudaMemcpy(dev_x, hst_x, N * sizeof(int), cudaMemcpyHostToDevice));
	cuda_exec(cudaMemcpy(dev_d, hst_d, M * sizeof(int), cudaMemcpyHostToDevice));

	block.x = atoi(argv[1]);
	grid.x = min((N + block.x - 1) / block.x, 65535);

	gpu_time -= timer();
	gpu_gather<<<grid, block>>>(dev_x, dev_d, dev_a, N);
	cuda_exec(cudaDeviceSynchronize());
	gpu_time += timer();

	cuda_exec(cudaMemcpy(hst_r, dev_a, N * sizeof(int), cudaMemcpyDeviceToHost));

	cpu_time -= timer();
	cpu_gather(hst_x, hst_d, hst_a, N);
	cpu_time += timer();

	check_result(hst_r, hst_a, N);

	printf("CPU time: %dms\n", (int) (1000 * cpu_time));
	printf("GPU time: %dms\n", (int) (1000 * gpu_time));
	
	free(hst_x);
	free(hst_d);
	free(hst_a);
	free(hst_r);
	
	cudaFree(dev_x);
	cudaFree(dev_d);
	cudaFree(dev_a);

	return 0;
}
