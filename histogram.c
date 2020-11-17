#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "proto.h"

#define ROOT 0
#define LEAF 1
#define MAX 257

void readInput(int *arr, int size);
int readSize();
void calculateHistogramOMP(int *histogram, int *numbers, int size);
void printHistogram(int *histogram, int *numbers, int size);
void mergeHistograms(int *histogram, int *other_histogram);

int main(int argc, char *argv[])
{
    int my_rank, num_procs, size, i, histogram[MAX], other_histogram[MAX], *all_numbers, *half_numbers;

    MPI_Comm matrix_comm;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Initialize histogram array
    for (i = 0; i < MAX; i++)
    {
        histogram[i] = 0;
    }

    if (my_rank == ROOT)
    {
        if (num_procs != 2)
        {
            printf("This program must be run with 2 MPI processes\n");
            exit(EXIT_FAILURE);
        }

        // Read amount of numbers to input from stdin
        size = readSize();
        MPI_Send(&size, 1, MPI_INT, LEAF, 0, MPI_COMM_WORLD);
        all_numbers = (int *)malloc(size * sizeof(int));

        // Read all numbers from stdin
        readInput(all_numbers, size);

        int remainder = size % 2;
        half_numbers = all_numbers + size / 2;

        // Send the second half of the numbers to the second MPI process
        MPI_Send(half_numbers, size / 2 + remainder, MPI_INT, LEAF, 0, MPI_COMM_WORLD);

        // Calculate half the numbers for the histogram using OMP
        calculateHistogramOMP(histogram, all_numbers, size / 2);

        // Receive the second half of the histogram from the second MPI process
        MPI_Recv(other_histogram, MAX, MPI_INT, LEAF, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Merge both histograms from both MPI processes into one histogram array
        mergeHistograms(histogram, other_histogram);

        printHistogram(histogram, all_numbers, size);

        // Free allocated all_numbers memory
        free(all_numbers);
    }
    else
    {
        // Receive the amount of numbers from the first MPI process
        MPI_Recv(&size, 1, MPI_INT, ROOT, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        int remainder = size % 2;
        half_numbers = (int *)malloc(size / 2 + remainder * sizeof(int));
        int *cuda_histogram = (int *)malloc(size / 2 + remainder * sizeof(int));

        // Receive half the numbers from the first MPI process
        MPI_Recv(half_numbers, size / 2 + remainder, MPI_INT, ROOT, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Calculate the second half of the numbers for the histogram using the CUDA and the GPU
        cuda_histogram = calculateHistogramCUDA(half_numbers, size / 2 + remainder);

        // Send the filled out histogram back to the first MPI process
        MPI_Send(cuda_histogram, MAX, MPI_INT, ROOT, 0, MPI_COMM_WORLD);

        // Free allocated cuda_histogram memory
        free(cuda_histogram);
    }

    MPI_Finalize();
    return 0;
}

void readInput(int *arr, int size)
{
    int i, input;

    printf("The amount of numbers to be inputted is: %d\n", size);
    printf("Enter integers(1-256) separated by WHITESPACE to fill the histogram\n");

    for (i = 0; i < size; i++)
    {
        scanf("%d", &input);
        if (input < 1 || input > 256)
        {
            printf("Incorrect input with number: %d. Input numbers must be between 1-256\n", input);
            exit(EXIT_FAILURE);
        }
        arr[i] = input;
    }
}

int readSize()
{
    int size;
    printf("Enter the amount of numbers.\n");
    scanf("%d", &size);
    if (size < 1)
    {
        printf("Incorrect amount of numbers: %d. Amount must be larger than 0\n", size);
        exit(EXIT_FAILURE);
    }
    return size;
}

void calculateHistogramOMP(int *histogram, int *numbers, int size)
{
#pragma omp parallel for
    for (int i = 0; i < size; i++)
    {
#pragma omp atomic
        histogram[numbers[i]] += 1;
    }
}

void mergeHistograms(int *histogram, int *other_histogram)
{
    for (int i = 1; i < MAX; i++)
        if (other_histogram[i] > 0)
            histogram[i] += other_histogram[i];
}

void printHistogram(int *histogram, int *numbers, int size)
{
    printf("\n====== Histogram ======\n");
    for (int i = 1; i < MAX; i++)
        if (histogram[i] > 0)
            printf("%d: %d\n", i, histogram[i]);
}
