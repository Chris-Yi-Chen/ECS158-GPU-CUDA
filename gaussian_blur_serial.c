#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>

#define MAX_LINE_LENGTH 1024
#define M_PI 3.14159265358979323846

// gcc -g -Wall -Wextra -Werror -O2 ./gaussian_blur_serial.c -o gaussian_blur_serial -lm
//./gaussian_blur_serial city_256.pgm city_256_1.pgm 1
//./ref_gaussian_blur_serial city_256.pgm city_256_ref.pgm 1
//compare -fuzz 1% -metric AE ./city_256_1.pgm city_256_ref.pgm null:

int calc_coord(int ind, int k_ind, int order, int length) {
    if (ind + k_ind - (order / 2) < 0) {
        return 0;
    } else if (ind + k_ind - (order / 2) >= length) {
        return length - 1;
    }
    
    return ind + k_ind - (order / 2);
}

void gaussian_blur(int *img, int *outImg, float* kernel, int width, int height, int order) {

    // x-axis
    for (int i = 0; i < width; i++) {
        // y-axis
        for (int j = 0; j < height; j++) {
            float sum = 0;
            int i_ind, j_ind;

            // iterate through kernel matrix and multiply weights
            // x-axis of kernel
            for (int ki = 0; ki < order; ki++) {
                i_ind = calc_coord(i, ki, order, width);
                // y-axis of kernel
                for (int kj = 0; kj < order; kj++) {
                    j_ind = calc_coord(j, kj, order, height);
                    sum += img[j_ind * width + i_ind] * kernel[kj * order + ki];
                }
            }
            //printf("%.8f ", sum);
            outImg[j * width + i] = (int)sum;
        }

        //printf("\n");
    }

}

float gaussian_func(int x, int y, float sigma) {
    /* FIX ME */
    // might need to normalize it so the sum = 1.0
    float exponent = -(x*x + y*y) / (2.0 * sigma * sigma);
    float e = exp(exponent);

    return (1.0 / (2.0 * M_PI * sigma * sigma)) * e;
}

void create_kernel_matrix(float **kernel, int order, float sigma) {
    *kernel = aligned_alloc(64, order * order * sizeof(float));
    int i, j;

    for (i = (int)(-order / 2); i <= (order / 2); i++) {
        for (j = (int)(-order / 2); j <= (order / 2); j++) {
            (*kernel)[(j + (order/2)) * order + (i + (order/2))] = gaussian_func(i, j, sigma);
            // printf("%.8f ", (*kernel)[j * order + i]);
        }
        // printf("\n");
    }    
}

void write_pgm(char *filename, int* map, size_t width, size_t height, int max_gray) {
    FILE* fp;
    size_t i;
    char* pixels;

    pixels = (char*)malloc(width * height);

    for (i = 0; i < width * height; i++) {
        pixels[i] = map[i];
    }

    /* Open file */
	fp = fopen(filename, "wb");
	if (!fp) {
		fprintf(stderr, "Error: cannot open file %s", filename);
		exit(1);
	}
    fprintf(fp, "P5\n%ld %ld\n%d\n", width, height, max_gray);
    fwrite(pixels, sizeof(char), width * height, fp);

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

    // fprintf(stderr, "%s, %s, %f\n", inFilename, outFilename, sigma);
    // fprintf(stderr, "%d x %d, %d\n", width, height, max_gray);

    /* Create Kernel Matrix */
    int order = (int)ceil(6.0 * sigma) % 2 == 1 ? ceil(6.0 * sigma) : ceil(6.0 * sigma) + 1; 
    create_kernel_matrix(&kernel, order, sigma);


    /* Call implementation */
    int *outImg = aligned_alloc(64, width * height * sizeof(int));
    gaussian_blur(img, outImg, kernel, width, height, order); 

    /* Save output image */
    write_pgm(outFilename, outImg, width, height, max_gray);

    /* Free resources */
    free(img);

    return 0;
}