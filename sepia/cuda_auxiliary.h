#pragma once

#include	<stdio.h>
#include	<stdlib.h>
#include	<sys/time.h>
#include	<time.h>


#define	open_file(fid, name, perm)			do {																					\
												if ((fid = fopen(name, perm)) == NULL) {											\
													fprintf(stderr, "%s: error opening file %s\n", argv[0], name);					\
													exit(EXIT_FAILURE);																\
												}																					\
											} while (0)

#define	close_file(fid)						do {																					\
												fid == NULL ? : fclose(fid);														\
											} while (0)

#define host_alloc(ptr, type, size)			do {																					\
												if ((ptr = (type *) malloc(size)) == NULL) {										\
													fprintf(stderr, "%s: error allocating memory\n", argv[0]);						\
													goto die;																		\
												}																					\
											} while (0)

#define	write_file(ptr, sz, cnt, fid, name)	do {																					\
												if (fwrite((ptr), (sz), (cnt), (fid)) != (cnt)) {									\
													fprintf(stderr, "%s: error writing to file %s\n", argv[0], (name));				\
											   		goto die;																		\
											 	}																					\
											} while (0)

#define read_file(ptr, sz, cnt, fid, name)	do {																					\
												if (fread((ptr), (sz), (cnt), (fid)) != (cnt)) {									\
													fprintf(stderr, "%s: error reading from file %s\n", argv[0], (name));			\
													goto die;																		\
												}																					\
											} while (0)

#define	cuda_exec(func_call)				do {																					\
												cudaError_t	error = (func_call);													\
																																	\
												if (error != cudaSuccess) {															\
													fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error));	\
													goto die;																		\
												}																					\
											} while (0)


double	timer()
{
	struct	timeval		tp;

	gettimeofday(&tp, NULL);

	return ((double) tp.tv_sec + (double) tp.tv_usec * 1.0e-6);
}

void	init_matrix(double *m, int nx, int ny, int lda)
{
	srand(time(0));

	for (int i = 0; i < ny; ++i) {
		for (int j = 0; j < nx; ++j)
//			m[j] = (double) rand() / RAND_MAX;
			m[j] = (double) 1.0;

		m += lda;
	}
}

