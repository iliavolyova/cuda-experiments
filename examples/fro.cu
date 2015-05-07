// nvcc fro.cu -o fro -lcublas -llapack
// računa Frobeniusovu normu n x n matrice pohranjene binarno po stupcima u
// datoteci A.dat

#include	<stdio.h>
#include	"cuda_auxiliary.h"

// za korištenje CUBLASa potrebno je koristiti header cublas_v2.h -- već uključujen
// u cuda_auxiliary.h

extern "C" double dlange_(const char *norm, int *M, int *N, double *A, int *LDA, double *work);

int main(int argc, char **argv)
{
	FILE			*fp = NULL;

	double			*hst_A = NULL;		
	double			*dev_A = NULL;
	double			*dev_N = NULL;		// norme stupaca	

	double			gpu_nrm;			// konačni rezultat
	double			cpu_nrm;			// konačni rezultat

	int				n;					// dimenzija matrice
	int				lda;				// vodeća dimenzija matrice A pohranjene na deviceu
	size_t			pitch;				

	double			total_time = 0.0;

	cublasHandle_t	handle;				// cublasHandle_t sadrži CUBLASov kontekst


	if (argc != 3) {
		fprintf(stderr, "usage: %s N A.dat\n", argv[0]);
		exit(EXIT_FAILURE);
	}

	if ((n = atoi(argv[1])) <= 0) {
		fprintf(stderr, "%s: matrix dimensions must be positive\n", argv[0]);
		exit(EXIT_FAILURE);
	}


	open_file(fp, argv[2], "r");
	host_alloc(hst_A, double, n * n);

	// alociranu memoriju na hostu deklariramo kao page-locked memoriju -- kopiranje bi trebalo biti brže
	cuda_exec(cudaHostRegister(hst_A, n * n * sizeof(double), cudaHostRegisterDefault));	
	read_file(hst_A, sizeof(double), n * n, fp);


	cuda_exec(cudaMalloc(&dev_N, n * sizeof(double)));
	cuda_exec(cudaMallocPitch(&dev_A, &pitch, n * sizeof(double), n));
	lda = pitch / sizeof(double);


	cublas_exec(cublasCreate(&handle));
	// pointer mode nam govori hoće li se parametri čitati s hosta ili devicea
	// detalji oko toga gdje neka funkcija učitava svoje parametre proučite u CUBLAS_Library.pdf
	cublas_exec(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));	
	// kopira matricu na device
	cublas_exec(cublasSetMatrix(n, n, sizeof(double), hst_A, n, dev_A, lda));


	total_time -= timer();

	for (int i = 0; i < n; ++i) {
		cublas_exec(cublasDnrm2(handle, n, dev_A + i * lda, 1, dev_N + i));
	}

	total_time += timer();

	// želimo rezultat zapisati na host -- CUBLAS implicitno kopira podatke
	cublas_exec(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));	
	// računamo Frobeniusovu normu vektora normi stupaca
	cublas_exec(cublasDnrm2(handle, n, dev_N, 1, &gpu_nrm));

	cpu_nrm = dlange_("F", &n, &n, hst_A, &n, hst_A);

	printf("CPU norm: %#.16lg\n", cpu_nrm);
	printf("CPU norm: %#.16lg\n", gpu_nrm);
	printf("Total time: %dms\n", (int) (1000 * total_time));


	cublas_exec(cublasDestroy(handle));

	cudaFree(dev_A);
	cudaFree(dev_N);

	cudaHostUnregister(hst_A);
	host_free(hst_A);

	return 0;
}
