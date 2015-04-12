#include <stdio.h>
#include "cuda_auxiliary.h"


__global__ void div1(float *c)
{
    unsigned int	tid = blockIdx.x * blockDim.x + threadIdx.x;
    float			ia = 0.0f;
	float			ib = 0.0f;

    if (tid % 2 == 0) {
        ia = 100.0f;
    } else {
        ib = 200.0f;
    }

    c[tid] = ia + ib;
}

__global__ void div2(float *c)
{
    unsigned int	tid = blockIdx.x * blockDim.x + threadIdx.x;
    float			ia = 0.0f;
	float			ib = 0.0f;

    if ((tid / warpSize) % 2 == 0) {
        ia = 100.0f;
    } else {
        ib = 200.0f;
    }

    c[tid] = ia + ib;
}

__global__ void div3(float *c)
{
    unsigned int	tid = blockIdx.x * blockDim.x + threadIdx.x;
    float			ia = 0.0f;
	float			ib = 0.0f;
    bool			ipred = (tid % 2 == 0);

    if (ipred) {
        ia = 100.0f;
    }

    if (!ipred) {
        ib = 200.0f;
    }

    c[tid] = ia + ib;
}

__global__ void div4(float *c)
{
    unsigned int	tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int	itid = tid >> 5;
    float			ia = 0.0f;
	float			ib = 0.0f;

    if (itid & 0x01 == 0) {
        ia = 100.0f;
    } else {
        ib = 200.0f;
    }

    c[tid] = ia + ib;
}

int main(int argc, char **argv)
{
	float	*dev_c = NULL;
	int		size;

	dim3	block;
	dim3	grid;

	double	exec_time[4] = {0};


	if (argc != 3) {
		fprintf(stderr, "usage: %s dimx num_elem\n", argv[0]);
		goto die;
	}

	size = atoi(argv[2]);

	block.x = atoi(argv[1]);
	grid.x = ((size + block.x - 1) / block.x);

	printf("Execution configuration: %d blocks, %d threads\n", grid.x, block.x);

    cuda_exec(cudaMalloc(&dev_c, size * sizeof(float)));

	
    exec_time[0] -= timer();
    div1<<<grid, block>>>(dev_c);
    cuda_exec(cudaDeviceSynchronize());
	exec_time[0] += timer();


    exec_time[1] -= timer();
    div2<<<grid, block>>>(dev_c);
    cuda_exec(cudaDeviceSynchronize());
	exec_time[1] += timer();


    exec_time[2] -= timer();
    div3<<<grid, block>>>(dev_c);
    cuda_exec(cudaDeviceSynchronize());
	exec_time[2] += timer();

    exec_time[3] -= timer();
    div4<<<grid, block>>>(dev_c);
    cuda_exec(cudaDeviceSynchronize());
	exec_time[3] += timer();

    printf("kernel div1 <<<%4d, %4d>>> elapsed %3dms\n", grid.x, block.x, (int) (1000 * exec_time[0]));
    printf("kernel div2 <<<%4d, %4d>>> elapsed %3dms\n", grid.x, block.x, (int) (1000 * exec_time[1]));
    printf("kernel div3 <<<%4d, %4d>>> elapsed %3dms\n", grid.x, block.x, (int) (1000 * exec_time[2]));
    printf("kernel div4 <<<%4d, %4d>>> elapsed %3dms\n", grid.x, block.x, (int) (1000 * exec_time[3]));

die:
	cuda_exec(cudaFree(dev_c));

    return 0;
}
