#include	"cuda_auxiliary.h"

__device__ int x[128];

__global__ void kernel(void)
{
	x[threadIdx.x] += threadIdx.x;
}

int main()
{
	int	hst_x[128];
	int	*ptr;

	for (int i = 0; i < 128; ++i)
		hst_x[i] = i;

	cudaMemcpyToSymbol(x, hst_x, 128 * sizeof(int));
	
	kernel<<<1,128>>>();
	cuda_exec(cudaDeviceSynchronize());

	// kopira u hst_x[0] peti element vektora x
	cudaMemcpyFromSymbol(hst_x, x, sizeof(int), 5 * sizeof(int));

	printf("%d\n\n", hst_x[0]);
	
	// u ptr dohvaćamo adresu na kojoj počinje polje x
	cudaGetSymbolAddress((void **) &ptr, x);

	cudaMemcpy(hst_x, ptr, 128 * sizeof(int), cudaMemcpyDeviceToHost);
	
	for (int i = 0; i < 128; ++i)
		printf("%d\n", hst_x[i]);

	return 0;
}
