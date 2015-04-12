#include	<stdio.h>

//compile: nvcc hello.cu -o hello

__global__	void print_hello_world(void)
{
	printf("Hello world from GPU\n");
}

int main(int argc, char **argv)
{
	printf("Hello world from CPU!\n");

	print_hello_world<<<2, 5>>>();
	cudaDeviceSynchronize();

	return 0;
}
