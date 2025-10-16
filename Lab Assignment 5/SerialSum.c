#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {
    // How many numbers to work with
    const long COUNT = 10000000;
    
    // The array to hold the numbers
    double *numbers;
    
    // Variables for the calculation
    double total = 0.0;
    double avg = 0.0;

    // Variables for timing
    clock_t t1, t2;

    // Get memory for our array
    numbers = (double*)malloc(COUNT * sizeof(double));
    if (numbers == NULL) {
        printf("Memory allocation failed.\n");
        return 1;
    }

    // --- Start the work ---
    t1 = clock(); // Start the timer

    // 1. Fill the array with numbers from 1 to COUNT
    for (long i = 0; i < COUNT; i++) {
        numbers[i] = i + 1;
    }

    // 2. Add up all the numbers in the array
    for (long i = 0; i < COUNT; i++) {
        total += numbers[i];
    }

    t2 = clock(); // Stop the timer
    // --- Work is done ---

    // Calculate the average and the time taken
    avg = total / COUNT;
    double time_taken = ((double)(t2 - t1)) / CLOCKS_PER_SEC;

    // Print the results
    printf("Sum: %.0f\n", total);
    printf("Average: %.2f\n", avg);
    printf("Time taken: %f seconds\n", time_taken);

    // Free up the memory we used
    free(numbers);
    
    return 0;
}