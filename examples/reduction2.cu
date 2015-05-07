// redukcija se ne bi smjela ovako kodirati

#include	<stdio.h>
#include	<float.h>
#include	"cuda_auxiliary.h"

__global__ void reduction(double *in_data, double *out_data, int N)
{
	double	*data = in_data + (blockIdx.x * blockDim.x << 1);

	for (int stride = 1; stride <= blockDim.x; stride <<= 1) {
		int index = (stride * threadIdx.x << 1);

		if (index < (blockDim.x << 1))
			data[index] += data[index + stride];

		__syncthreads();

		index <<= 1;
	}
		
	if (threadIdx.x == 0)
		out_data[blockIdx.x] = data[0];
}

double sum_array(double *data, int size)
{
	double	sum = 0.0;

	for (int i = 0; i < size; ++i)
		sum += data[i];

	return sum;
}	

int	main(int argc, char **argv)
{
	double		*dev_in_array = NULL;
	double		*dev_out_array = NULL;
	double		*hst_tmp = NULL;
	double		*hst_array = NULL;

	int			N = (1 << 20); 

	double		cpu_sum;
	double		gpu_sum;

	dim3		grid;
	dim3		block;

	double		cpu_time = 0.0;
	double		gpu_time = 0.0;


	if (argc != 2) {
		fprintf(stderr, "usage: %s dimx\n", argv[0]);
		goto die;
	}

	host_alloc(hst_array, double, N * sizeof(double));
	host_alloc(hst_tmp, double, N * sizeof(double));
	init_matrix(hst_array, N, 1, 0);

	cuda_exec(cudaMalloc(&dev_in_array, N * sizeof(double)));
	cuda_exec(cudaMalloc(&dev_out_array, N * sizeof(double)));
	cuda_exec(cudaMemcpy(dev_in_array, hst_array, N * sizeof(double), cudaMemcpyHostToDevice));

	block.x = atoi(argv[1]);
	grid.x = ((N + block.x - 1) / block.x) / 2;

	cpu_time -= timer();
	cpu_sum = sum_array(hst_array, N);
	cpu_time += timer();

	gpu_time -= timer();
	reduction<<<grid, block>>>(dev_in_array, dev_out_array, N);
	cudaDeviceSynchronize();
	gpu_time += timer();

	cuda_exec(cudaMemcpy(hst_tmp, dev_out_array, grid.x * sizeof(double), cudaMemcpyDeviceToHost));

	gpu_time -= timer();
	gpu_sum = 0.0;

	for (int i = 0; i < grid.x; ++i)
		gpu_sum += hst_tmp[i];
	gpu_time += timer();

	printf("CPU sum: %#.16lg\n", cpu_sum);
	printf("GPU sum: %#.16lg\n", gpu_sum);
	printf("Execution configuration: %d blocks, %d threads\n", grid.x, block.x);
	printf("CPU execution time: %#.3lgs\n", cpu_time);
	printf("GPU execution time: %#.3lgs\n", gpu_time);

die:
	cuda_exec(cudaFree(dev_in_array));
	cuda_exec(cudaFree(dev_out_array));

	free(hst_array);
	
	return 0;
}
	
	
