#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define ROOT 0
#define MAX 257

void readInput(int *arr, int size);
int readSize();
void printHistogram(int *histogram, int *numbers, int size);

int main(int argc, char *argv[])
{
    int my_rank, num_procs, size, i, histogram[MAX], *numbers;

    MPI_Comm matrix_comm;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    if (my_rank == ROOT)
    {
        // Initialize histogram array
        for (i = 0; i < MAX; i++)
        {
            histogram[i] = 0;
        }

        size = readSize();
        numbers = (int *)malloc(size * sizeof(int));

        readInput(numbers, size);
        printHistogram(histogram, numbers, size);
        // free allocated memory
        free(numbers);
    }

    MPI_Finalize();
    return 0;
}

void readInput(int *arr, int size)
{
    int i = 0, input;

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

void printHistogram(int *histogram, int *numbers, int size)
{
    int i;
    for (i = 0; i < size; i++)
    {
        histogram[numbers[i]] += 1;
    }
    printf("\n====== Histogram ======\n");
    for (i = 0; i < MAX; i++)
    {
        if (histogram[i] > 0)
        {
            printf("%d: %d\n", i, histogram[i]);
        }
    }
}
