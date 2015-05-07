#include	<stdio.h>
#include	<stdlib.h>
#include	"cuda_auxiliary.h"
#include	"CImg.h"

using namespace cimg_library;


__global__ void gpu_sepia(unsigned char *data, int width, int height, int lda)
{
	const int ix = blockDim.x * blockIdx.x + threadIdx.x;
	const int iy = blockDim.y * blockIdx.y + threadIdx.y;

	uchar4 r[2];
	uchar4 g[2];
	uchar4 b[2];

	uchar4 nr;
	uchar4 ng;
	uchar4 nb;

	if (iy >= height)
		return;

	r[0] = *((uchar4 *) (data + (0 * height + iy) * lda) + ix);
	g[0] = *((uchar4 *) (data + (1 * height + iy) * lda) + ix);
	b[0] = *((uchar4 *) (data + (2 * height + iy) * lda) + ix);
	
	r[1] = *((uchar4 *) (data + (0 * height + iy) * lda) + ix + blockDim.x * gridDim.x);
	g[1] = *((uchar4 *) (data + (1 * height + iy) * lda) + ix + blockDim.x * gridDim.x);
	b[1] = *((uchar4 *) (data + (2 * height + iy) * lda) + ix + blockDim.x * gridDim.x);

	nr.x = min(0.393f * r[0].x + 0.769f * g[0].x + 0.189f * b[0].x, 255.0f);
	nr.y = min(0.393f * r[0].y + 0.769f * g[0].y + 0.189f * b[0].y, 255.0f);
	nr.z = min(0.393f * r[0].z + 0.769f * g[0].z + 0.189f * b[0].z, 255.0f);
	nr.w = min(0.393f * r[0].w + 0.769f * g[0].w + 0.189f * b[0].w, 255.0f);

	ng.x = min(0.349f * r[0].x + 0.686f * g[0].x + 0.168f * b[0].x, 255.0f);
	ng.y = min(0.349f * r[0].y + 0.686f * g[0].y + 0.168f * b[0].y, 255.0f);
	ng.z = min(0.349f * r[0].z + 0.686f * g[0].z + 0.168f * b[0].z, 255.0f);
	ng.w = min(0.349f * r[0].w + 0.686f * g[0].w + 0.168f * b[0].w, 255.0f);

	nb.x = min(0.272f * r[0].x + 0.534f * g[0].x + 0.131f * b[0].x, 255.0f);
	nb.y = min(0.272f * r[0].y + 0.534f * g[0].y + 0.131f * b[0].y, 255.0f);
	nb.z = min(0.272f * r[0].z + 0.534f * g[0].z + 0.131f * b[0].z, 255.0f);
	nb.w = min(0.272f * r[0].w + 0.534f * g[0].w + 0.131f * b[0].w, 255.0f);

	*((uchar4 *) (data + (0 * height + iy) * lda) + ix) = nr;
	*((uchar4 *) (data + (1 * height + iy) * lda) + ix) = ng;
	*((uchar4 *) (data + (2 * height + iy) * lda) + ix) = nb;

	nr.x = min(0.393f * r[1].x + 0.769f * g[1].x + 0.189f * b[1].x, 255.0f);
	nr.y = min(0.393f * r[1].y + 0.769f * g[1].y + 0.189f * b[1].y, 255.0f);
	nr.z = min(0.393f * r[1].z + 0.769f * g[1].z + 0.189f * b[1].z, 255.0f);
	nr.w = min(0.393f * r[1].w + 0.769f * g[1].w + 0.189f * b[1].w, 255.0f);

	ng.x = min(0.349f * r[1].x + 0.686f * g[1].x + 0.168f * b[1].x, 255.0f);
	ng.y = min(0.349f * r[1].y + 0.686f * g[1].y + 0.168f * b[1].y, 255.0f);
	ng.z = min(0.349f * r[1].z + 0.686f * g[1].z + 0.168f * b[1].z, 255.0f);
	ng.w = min(0.349f * r[1].w + 0.686f * g[1].w + 0.168f * b[1].w, 255.0f);

	nb.x = min(0.272f * r[1].x + 0.534f * g[1].x + 0.131f * b[1].x, 255.0f);
	nb.y = min(0.272f * r[1].y + 0.534f * g[1].y + 0.131f * b[1].y, 255.0f);
	nb.z = min(0.272f * r[1].z + 0.534f * g[1].z + 0.131f * b[1].z, 255.0f);
	nb.w = min(0.272f * r[1].w + 0.534f * g[1].w + 0.131f * b[1].w, 255.0f);

	*((uchar4 *) (data + (0 * height + iy) * lda) + ix + blockDim.x * gridDim.x) = nr;
	*((uchar4 *) (data + (1 * height + iy) * lda) + ix + blockDim.x * gridDim.x) = ng;
	*((uchar4 *) (data + (2 * height + iy) * lda) + ix + blockDim.x * gridDim.x) = nb;

}

void hst_sepia(unsigned char *data, int width, int height)
{
	unsigned char *r = data + 0 * width * height;
	unsigned char *g = data + 1 * width * height;
	unsigned char *b = data + 2 * width * height;
	
	unsigned char nr;
	unsigned char ng;
	unsigned char nb;
	
	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			nr = min(0.393f * r[j] + 0.769f * g[j] + 0.189f * b[j], 255.0f);
			ng = min(0.349f * r[j] + 0.686f * g[j] + 0.168f * b[j], 255.0f);
			nb = min(0.272f * r[j] + 0.534f * g[j] + 0.131f * b[j], 255.0f);

			r[j] = nr;
			g[j] = ng;
			b[j] = nb;
		}

		r += width;
		g += width;
		b += width;
	}
}
int main(int argc, char **argv)
{
	CImg<unsigned char>	cpu_image;
	CImg<unsigned char>	gpu_image;

	unsigned char		*hst_data = NULL;
	unsigned char		*dev_data = NULL;

	size_t				pitch;
	int					lda;

	int					width;
	int					height;
	dim3				block;
	dim3				grid;

	double				cpu_time = 0.0;
	double				gpu_time = 0.0;


	if (argc != 5) {
		fprintf(stderr, "usage: %s dimx dimy in_image out_image\n", argv[0]);
		exit(EXIT_FAILURE);
	}


	cpu_image = CImg<unsigned char>(argv[3]);
	gpu_image = CImg<unsigned char>(cpu_image.width(), cpu_image.height(), 1, 3);

	hst_data	= cpu_image.data();
	width		= cpu_image.width();
	height		= cpu_image.height();


	block.x = atoi(argv[1]);
	block.y = atoi(argv[2]);

	grid.x = min((width + 8 * block.x - 1) / (8 * block.x), 65535);
	grid.y = min((height + block.y - 1) / block.y, 65535);


	cuda_exec(cudaMallocPitch(&dev_data, &pitch, width * sizeof(unsigned char), 3 * height));
	cuda_exec(cudaMemcpy2D(dev_data, pitch, hst_data, width * sizeof(unsigned char), width * sizeof(unsigned char), 3 * height, cudaMemcpyHostToDevice));

	lda = pitch / sizeof(unsigned char);

	gpu_time -= timer();
	gpu_sepia<<<grid, block>>>(dev_data, width, height, lda);
	cuda_exec(cudaDeviceSynchronize());
	gpu_time += timer();

	cuda_exec(cudaMemcpy2D(gpu_image.data(), width * sizeof(unsigned char), dev_data, pitch, width * sizeof(unsigned char), 3 * height, cudaMemcpyDeviceToHost));
	gpu_image.save_bmp(argv[4]);

	cpu_time -= timer();
	hst_sepia(cpu_image.data(), width, height);
	cpu_time += timer();

	for (int i = 0; i < 3 * height * width; ++i)
		if (cpu_image.data()[i] != gpu_image.data()[i])
			printf("Razlika na poziciji %d: %u %u\n", i, cpu_image.data()[i], gpu_image.data()[i]);
			
	printf("width = %d, height = %d, lda = %d\n", width, height, lda);
	printf("Execution configuration: %d x %d grid, %d x %d blocks\n", grid.x, grid.y, block.x, block.y);
	printf("CPU time: %dms\n", (int) (1000 * cpu_time));
	printf("GPU time: %dms\n", (int) (1000 * gpu_time));

	cuda_exec(cudaFree(dev_data));

	return 0;
}
