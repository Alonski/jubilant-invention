#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define ROOT 0
#define LEAF 1
#define MAX 257

void readInput(int *arr, int size);
int readSize();
void calculateHistogram(int *histogram, int *numbers, int size);
void printHistogram(int *histogram, int *numbers, int size);

int main(int argc, char *argv[])
{
    int my_rank, num_procs, size, i, histogram[MAX], *all_numbers, *half_numbers;

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
            exit(-1);
        }
        size = readSize();
        MPI_Send(&size, 1, MPI_INT, LEAF, 0, MPI_COMM_WORLD);
        all_numbers = (int *)malloc(size * sizeof(int));

        readInput(all_numbers, size);
        int remainder = size % 2;
        half_numbers = all_numbers + size / 2;
        MPI_Send(half_numbers, size / 2 + remainder, MPI_INT, LEAF, 0, MPI_COMM_WORLD);

        printf("Size/2: %d\n", size / 2);
        printf("Size/2 + remainder: %d\n", size / 2 + remainder);
        calculateHistogram(histogram, all_numbers, size / 2);
        printHistogram(histogram, all_numbers, size / 2);

        // free allocated memory
        free(all_numbers);
    }
    else
    {
        MPI_Recv(&size, 1, MPI_INT, ROOT, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        int remainder = size % 2;
        half_numbers = (int *)malloc(size / 2 + remainder * sizeof(int));
        MPI_Recv(half_numbers, size / 2 + remainder, MPI_INT, ROOT, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        calculateHistogram(histogram, half_numbers, size / 2 + remainder);
        printHistogram(histogram, half_numbers, size / 2 + remainder);
        // free allocated memory
        free(half_numbers);
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
            exit(-1);
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
        exit(-1);
    }
    return size;
}

void calculateHistogram(int *histogram, int *numbers, int size)
{
    int i;
    for (i = 0; i < size; i++)
    {
        histogram[numbers[i]] += 1;
    }
}

void printHistogram(int *histogram, int *numbers, int size)
{
    int i;
    printf("\n====== Histogram ======\n");
    for (i = 1; i < MAX; i++)
    {
        if (histogram[i] > 0)
        {
            printf("%d: %d\n", i, histogram[i]);
        }
    }
}
