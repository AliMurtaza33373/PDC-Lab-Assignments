#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[]) {
    // Basic MPI variables
    int my_rank, num_procs;

    const long TOTAL_COUNT = 10000000;
    long part_size; // How many numbers each process gets

    // Arrays
    double *all_numbers = NULL; // The full array, only on rank 0
    double *my_part;           // The chunk of numbers for this process

    // Sums
    double sum_part = 0.0;     // The sum of my part
    double total_sum = 0.0;    // The final result

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Calculate how big each chunk should be
    part_size = TOTAL_COUNT / num_procs;

    // Everyone gets memory for their own part
    my_part = (double*)malloc(part_size * sizeof(double));

    // The main process (rank 0) creates and fills the big array
    if (my_rank == 0) {
        all_numbers = (double*)malloc(TOTAL_COUNT * sizeof(double));
        // Fill it with 1, 2, 3, ..., N
        for (long i = 0; i < TOTAL_COUNT; i++) {
            all_numbers[i] = (double)(i + 1);
        }
    }

    // Scatter the big array from rank 0 out to everyone
    MPI_Scatter(all_numbers, part_size, MPI_DOUBLE,
                my_part, part_size, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // Now, every process adds up the numbers in its own part
    for (long i = 0; i < part_size; i++) {
        sum_part += my_part[i];
    }

    // Reduce all the partial sums into one final sum on rank 0
    MPI_Reduce(&sum_part, &total_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // Rank 0 prints the final result
    if (my_rank == 0) {
        // The real answer is N * (N+1) / 2
        double expected_sum = (double)TOTAL_COUNT * (TOTAL_COUNT + 1) / 2.0;

        printf("The final calculated sum is: %.0f\n", total_sum);
        printf("The expected math answer is: %.0f\n", expected_sum);
        printf("Difference: %f\n", expected_sum - total_sum);

        // Free the big array that only rank 0 used
        free(all_numbers);
    }

    // Everyone frees the memory for their part
    free(my_part);

    MPI_Finalize();
    return 0;
}