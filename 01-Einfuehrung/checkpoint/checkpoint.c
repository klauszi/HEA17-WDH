/* ************************************************************************ */
/* Include standard header file.                                            */
/* ************************************************************************ */

#define _XOPEN_SOURCE 500

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <malloc.h>
#include <errno.h>
#include <sys/time.h>
#include <time.h>
#include <omp.h>
#include <math.h>

#define N 360
#define DEBUG 1

/* time measurement variables */
struct timeval start_time;       /* time when program started               */
struct timeval comp_time;        /* time when calculation complet           */

/* ************************************************************************ */
/*  allocate matrix of size N x N                                           */
/* ************************************************************************ */
static
double**
alloc_matrix (void)
{
	double** matrix;
	int i;

	// allocate pointer array to the beginning of each row
	// rows
	matrix = malloc(N * sizeof(double*));

	if (matrix == NULL)
	{
		printf("Allocating matrix lines failed!\n");
		exit(EXIT_FAILURE);
	}

	// allocate memory for each row and assig it to the frist dimensions pointer
	// columns
	for (i = 0; i < N; i++)
	{
		matrix[i] = malloc(N * sizeof(double));

		if (matrix[i] == NULL)
		{
			printf("Allocating matrix elements failed!\n");
			exit(EXIT_FAILURE);
		}
	}

	return matrix;
}

/* ************************************************************************* */
/*  free matrix								                                 */
/* ************************************************************************* */
static void free_matrix (double **matrix)
{
	int i;

	// free memory for allocated for each row
	for (i = 0; i < N; i++)
	{
		free(matrix[i]);
	}
	// free pointer array to the beginning of each row
	free(matrix);
}

/* ************************************************************************* */
/*  init matrix									                             */
/* ************************************************************************* */
static
void
init_matrix (double** matrix)
{
	int i, j;

	srand(time(NULL));

	for (i = 0; i < N; i++)
	{
		for (j = 0; j < N; j++)
		{
			matrix[i][j] = 10 * j;
		}
	}
}

/* ************************************************************************* */
/*  init matrix reading file					         	                             */
/* ************************************************************************* */
static
void
read_matrix (double** matrix, const char* path, int* current_iteration)
{
    int line;
    int file_descriptor, read_error;
    int base_offset = 3 * sizeof(int);

    file_descriptor = open(path, O_RDONLY, 0600);
    read_error = pread(file_descriptor, current_iteration, sizeof(int), 2 * sizeof(int));
    if ( read_error == -1 )
    {
        printf("Read Error in Header through read_matrix");
        exit(EXIT_FAILURE);
    }

	for (line = 0; line < N; line++)
	{
            int read_block = sizeof (matrix) * N;
            int offset = read_block * line + base_offset;
            read_error = pread(file_descriptor, matrix[line], read_block, offset);
            if ( read_error == -1)
            {
                printf("Read Error in read_matrix: row %d", line);
                exit(EXIT_FAILURE);

            }
	}
    close(file_descriptor);
}

/* ************************************************************************ */
/*  displaydebug: displays test                             */
/* ************************************************************************ */
static
void
displaydebug (double** matrix)
{
    int i, j;
    int file_descriptor, read_error;;
    int base_offset;
    int equal = 1;
    double precision = 0.0000001;
    base_offset = 3 * sizeof(int);
    if(DEBUG)
    {
        double tmp_value;
        int tmp_header_value;


        // read data
        file_descriptor = open("matrix.out", O_RDONLY, 0600);
	    for (i = 0; i < N; i++)
	    {
	    	for (j = 0; j < N; j++)
	    	{
                int correct;
                read_error = pread(file_descriptor, &tmp_value, sizeof(double), sizeof(double) * (N * i + j) + base_offset);
                if ( read_error == -1)
                {
                    printf("Read Error: row %d and column %d", i, j);
                    exit(EXIT_FAILURE);
                }
                correct = fabs(tmp_value -  matrix[i][j]) < precision || (isnan(tmp_value) && isnan(matrix[i][j]));
                equal = correct && equal? 1 : 0; 
	    	}
	    }
        printf("DEBUGINFO: Matrix und matrix.out stimmen%süberein\n", equal? " " : " nicht ");
        close(file_descriptor);

        // read header
        file_descriptor = open("matrix.out", O_RDONLY, 0600);
        for (i = 0; i < 3; i++)
        {
            char headernames[3][20] = {"Threads", "Iterationen", "Letzte Iteration"};
            read_error = pread(file_descriptor, &tmp_header_value, sizeof(int), i * sizeof(int));
            if ( read_error == -1 )
            {
                printf("Read Error in Header");
                exit(EXIT_FAILURE);
            }
            printf("%s: %d\n", headernames[i], tmp_header_value);
        }
        close(file_descriptor);
 }
}

/* ************************************************************************ */
/*  calculate                                                               */
/* ************************************************************************ */
static
void
calculate (double** matrix, int iterations, int threads, int start_iteration)
{
	int i, j, k, l;
	int tid;
	int lines, from, to;
    int file_descriptor = open("temp_matrix.out", O_CREAT | O_WRONLY, 0600);
    int bytes_written;
    int base_offset = 3 * sizeof(int);

    if (file_descriptor == -1)
    {
        printf("temp_matrix.out cannot open. Exit");
        exit(0);
    }

	tid = 0;
	lines = from = to = -1;

	// Explicitly disable dynamic teams
	omp_set_dynamic(0);
	omp_set_num_threads(threads);

	#pragma omp parallel firstprivate(tid, lines, from, to) private(k, l, i, j)
	{
		tid = omp_get_thread_num();

		lines = (tid < (N % threads)) ? ((N / threads) + 1) : (N / threads);
		from =  (tid < (N % threads)) ? (lines * tid) : ((lines * tid) + (N % threads));
		to = from + lines;

		for (k = start_iteration; k <= iterations; k++)
		{
			for (i = from; i < to; i++)
			{
				for (j = 0; j < N; j++)
				{
					for (l = 1; l <= 4; l++)
					{
						matrix[i][j] = cos(matrix[i][j]) * sin(matrix[i][j]) * sqrt(matrix[i][j]) / tan(matrix[i][j]) / log(matrix[i][j]) * k * l;
					}
				}
			}

            if(tid == 0)
            {
                int header[] = {threads, iterations, k};
                bytes_written = pwrite(file_descriptor, header, sizeof(header), 0);
                if ( bytes_written < (int) sizeof (header) )
                {
                    printf("Header Write Error. Exit");
                    exit(EXIT_FAILURE);
                }
            }


	        for (i = from; i < to; i++)
		    {
                int write_block = sizeof (matrix) * N;
                int offset = write_block * i + base_offset;
                bytes_written = pwrite(file_descriptor, matrix[i], write_block, offset);
                if ( bytes_written < (int) sizeof (double) )
                {
                    printf("Write Error from Thread %d. Exit", tid);
                    exit(EXIT_FAILURE);
                }
		    }

			#pragma omp barrier


            if (tid == 0)
            {
                char oldname[] = "temp_matrix.out";
                char newname[] = "matrix.out";
                int ret = 1;
                close(file_descriptor);
                
                ret = rename(oldname, newname);
                 
                if(ret != 0)
                {
                   printf("Error: unable to rename the file");
                   exit(EXIT_FAILURE);
                }
                file_descriptor = open("temp_matrix.out", O_CREAT | O_WRONLY, 0600);
            }

			#pragma omp barrier
		}
        close(file_descriptor);

	}

    displaydebug(matrix);
}

/* ************************************************************************ */
/*  displayStatistics: displays some statistics                             */
/* ************************************************************************ */
static
void
displayStatistics (int iterations)
{
	double time = (comp_time.tv_sec - start_time.tv_sec) + (comp_time.tv_usec - start_time.tv_usec) * 1e-6;
	printf("Berechnungszeit: %fs\n", time);

	printf("Durchsatz:       %f MB/s\n", (double) iterations * N * N * sizeof(double) / time * 1e-6);

	printf("IOPS:            %f Op/s\n", (double) iterations * N / time);
}

/* ************************************************************************ */
/*  is_checkpoint_useful: check if matrix.out exists and if it is useful */
/* ************************************************************************ */
static
int
is_checkpoint_useful (int target_iterations)
{
    int file_descriptor, read_error;
    int checkpoint_target_iterations;

    if (access("matrix.out", F_OK) == -1)
    {
        return 0;
    }
    else
    {
        file_descriptor = open("matrix.out", O_RDONLY, 0600);
        read_error = pread(file_descriptor, &checkpoint_target_iterations, sizeof(int), 2 * sizeof(int));
        if ( read_error == -1 )
        {
            printf("Read Error in Header through is_checkpoint_useful");
            exit(EXIT_FAILURE);
        }
        close(file_descriptor);
        if ( checkpoint_target_iterations <= target_iterations ){
            return 1;
        }
        else
        {
            return 0;
        }
    }
}

/* ************************************************************************ */
/*  main                                                                    */
/* ************************************************************************ */
int
main (int argc, char** argv)
{
	int threads, iterations, current_iteration;
	double** matrix;

    current_iteration = 0;

	if (argc < 3)
	{
		printf("Usage: %s threads iterations\n", argv[0]);
		exit(EXIT_FAILURE);
	}
	else
	{
		sscanf(argv[1], "%d", &threads);
		sscanf(argv[2], "%d", &iterations);
	}

	matrix = alloc_matrix();

	if (is_checkpoint_useful(iterations))
	{
        printf("checkpoint is useful!\n");
		read_matrix(matrix, "matrix.out", &current_iteration);
	}
	else
	{
		init_matrix(matrix);
	}

	gettimeofday(&start_time, NULL);
	calculate(matrix, iterations, threads, current_iteration + 1);
	gettimeofday(&comp_time, NULL);

	displayStatistics(iterations);

	free_matrix(matrix);

	return 0;
}
