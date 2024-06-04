// ./gaussian_blur_cuda city_256.pgm city_256_1.pgm 1
// nvcc -Xcompiler -Wall -Xcompiler -Werror ... -o gaussian_blur_cuda gaussian_blur_cuda.cu -lm

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>

#define MAX_LINE_LENGTH 1024
#define M_PI 3.14159265358979323846
#define MAX_THREADS 1024

#define cuda_check(ret) _cuda_check((ret), __FILE__, __LINE__)
inline void _cuda_check(cudaError_t ret, const char *file, int line)
{
	if (ret != cudaSuccess) {
		fprintf(stderr, "CudaErr: %s (%s:%d)\n", cudaGetErrorString(ret), file, line);
		exit(1);
	}
}

__device__ int calc_coord(int ind, int k_ind, int order, int length) {
    if (ind + k_ind - (order / 2) < 0) {
        return 0;
    } else if (ind + k_ind - (order / 2) >= length) {
        return length - 1;
    }
    return ind + k_ind - (order / 2);
}

__global__ void gaussian_blur_kernel(int *img, int *outImg, float* kernel, int N, int order)
{
    // row = j, col = i
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    /* Discard out-of-bound coordinates */
    if (row >= N || col >= N) {
		return;
    }

    int i_ind, j_ind;
    float sum = 0;
    
    // x-axis of kernel
    for (int ki = 0; ki < order; ki++) {
        i_ind = calc_coord(col, ki, order, N);
        // y-axis of kernel
        for (int kj = 0; kj < order; kj++) {
            j_ind = calc_coord(row, kj, order, N);
            sum += img[j_ind * N + i_ind] * kernel[kj * order + ki];
        }
    }
    outImg[row * N + col] = (int)sum;
}
void gaussian_blur_cuda(int *img_h, int *outImg_h, float* kernel_h, int width, int height, int order)
{

    /* Memory Setup */
    int *img_d, *outImg_d;
    float *kernel_d;
    size_t size = height * width * sizeof(int);

    cuda_check(cudaMalloc(&img_d, size));
    cuda_check(cudaMalloc(&outImg_d, size));
    cuda_check(cudaMalloc(&kernel_d, order * order * sizeof(float)));

    cuda_check(cudaMemcpy(img_d, img_h, size, cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(outImg_d, outImg_h, size, cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(kernel_d, kernel_h, order * order * sizeof(float), cudaMemcpyHostToDevice));



    /* Invoke Kernel function */
    int grid = ceil(width / 32.0);
    // fprintf(stderr, "%d\n", grid);

    dim3 grid_dim(grid);
    dim3 block_dim(32, 32);
    gaussian_blur_kernel<<<grid_dim, block_dim>>>(img_d, outImg_d, kernel_h, width, order);

    cuda_check(cudaPeekAtLastError());      /* Catch configuration errors */
	cuda_check(cudaDeviceSynchronize());    /* Catch execution errors */

    /* Copy outImg from device memory */
	cuda_check(cudaMemcpy(outImg_h, outImg_d, size, cudaMemcpyDeviceToHost));

    /* Free Device Memory */
    cuda_check(cudaFree(img_d));
    cuda_check(cudaFree(outImg_d));
    cuda_check(cudaFree(kernel_d));

}
/* 
1. Memory setup, data prep
2. Invoke the kernel
3. Retrieve results, memory cleanup
*/

float gaussian_func(int x, int y, float sigma) {
    /* FIX ME */
    // might need to normalize it so the sum = 1.0
    float exponent = -(x*x + y*y) / (2.0 * sigma * sigma);
    float e = exp(exponent);

    return (1.0 / (2.0 * M_PI * sigma * sigma)) * e;
}

void create_kernel_matrix(float **kernel, int order, float sigma) {
    *kernel = (float*)aligned_alloc(64, order * order * sizeof(float));
    int i, j;

    for (i = (int)(-order / 2); i <= (order / 2); i++) {
        for (j = (int)(-order / 2); j <= (order / 2); j++) {
            (*kernel)[(j + (order/2)) * order + (i + (order/2))] = gaussian_func(i, j, sigma);
        }
    }    
}

void write_pgm(char *filename, int* map, size_t N, int max_gray) {
    FILE* fp;
    size_t i;
    char* pixels;

    pixels = (char*)malloc(N * N);

    for (i = 0; i < N * N; i++) {
        pixels[i] = map[i];
    }

    /* Open file */
	fp = fopen(filename, "wb");
	if (!fp) {
		fprintf(stderr, "Error: cannot open file %s", filename);
		exit(1);
	}
    fprintf(fp, "P5\n%ld %ld\n%d\n", N, N, max_gray);
    fwrite(pixels, sizeof(char), N * N, fp);

    free(pixels);
    fclose(fp);

}

void read_pgm(const char *filename, int *width, int *height, int *max_gray, int **img) {
    int i;
    FILE *file = fopen(filename, "r");
    char line[MAX_LINE_LENGTH];
    // Read width and height, skipping comments
    while (fgets(line, sizeof(line), file)) {
        if (line[0] == '#') continue; // Skip comments

        if (sscanf(line, "%d %d", width, height) == 2) break;
    }

    while (fgets(line, sizeof(line), file)) {
        if (line[0] == '#') continue; // Skip comments

        if (sscanf(line, "%d", max_gray) == 1) break;
    }
    
    *img = (int*)aligned_alloc(64, (*width) * (*height) * sizeof(int));
    if (!(*img)) {
        perror("img alloc failed");
        fclose(file);
        exit(1);
    }

    for (i = 0; i < (*width) * (*height); i++) {
        int pixel = fgetc(file);
        if (pixel == EOF) {
            fprintf(stderr, "Unexpected end of file\n");
            free(*img);
            fclose(file);
            return;
        }
        (*img)[i] = pixel;
    }



    fclose(file);
}

void parse_float(char *str, char *val, float min, float max, float *num)
{
	float n = atof(str);
	if (n < min || n > max) {
		fprintf(stderr, "Error: wrong %s (%lf <= N <= %lf)", val, min, max);
		exit(1);
	}
	*num = n;
}

int main(int argc, char *argv[])
{
	float sigma;
    float *kernel;
    int width, height, max_gray;
    int *img;
    char *inFilename, *outFilename;

	/* Command line arguments */
	if (argc < 4) {
		fprintf(stderr, "Usage: %s <input_pgm> <output_pgm> <sigma>\n",
				argv[0]);
		exit(1);
	}

    inFilename = argv[1];
    read_pgm(inFilename, &width, &height, &max_gray, &img);
    outFilename = argv[2];
	parse_float(argv[3], "sigma",  0, 10, &sigma);

    fprintf(stderr, "%d, %d\n", width, height);

    /* Create Kernel Matrix */
    int order = (int)ceil(6.0 * sigma) % 2 == 1 ? ceil(6.0 * sigma) : ceil(6.0 * sigma) + 1; 
    create_kernel_matrix(&kernel, order, sigma);


    /* Call implementation */
    int *outImg = (int *)aligned_alloc(64, width * height * sizeof(int));
    gaussian_blur_cuda(img, outImg, kernel, width, height, order); 

    /* Save output image */
    write_pgm(outFilename, outImg, height, max_gray);

    /* Free resources */
    free(img);

    return 0;
}