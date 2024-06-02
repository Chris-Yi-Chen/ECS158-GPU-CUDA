// ./gaussian_blur_cuda city_256.pgm city_256_1.pgm 1
// nvcc -Xcompiler -Wall -Xcompiler -Werror ... -o gaussian_blur_cuda gaussian_blur_cuda.cu -lm

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>

#define MAX_LINE_LENGTH 1024
#define M_PI 3.14159265358979323846

#define cuda_check(ret) _cuda_check((ret), __FILE__, __LINE__)
inline void _cuda_check(cudaError_t ret, const char *file, int line)
{
	if (ret != cudaSuccess) {
		fprintf(stderr, "CudaErr: %s (%s:%d)\n", cudaGetErrorString(ret), file, line);
		exit(1);
	}
}


void gaussian_blur_cuda(int *img_h, int *outImg_h, float* kernel_h, int width, int height, int order) {

    dim3 grid_dim(1);
    dim3 block_dim(N, N);

}
/* 
1. Memory setup, data prep
2. Invoke the kernel
3. Retrieve results, memory cleanup
*/
void create_kernel_matrix(float **kernel, int order, float sigma) {
    *kernel = aligned_alloc(64, order * order * sizeof(float));
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

    pixels = malloc(N * N);

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
    
    *img = aligned_alloc(64, (*width) * (*height) * sizeof(int));
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