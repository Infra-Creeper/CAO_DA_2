#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

// Function to integrate (you can change this)
double f(double x) {
    return x * x;  // Example: f(x) = x^2
}

// Riemann sum using left endpoint method
double riemann_sum_left(double a, double b, long n) {
    double delta = (b - a) / n;  // Width of each rectangle
    double sum = 0.0;
    
    #pragma omp parallel for reduction(+:sum)
    for (long i = 0; i < n; i++) {
        double x = a + i * delta;
        sum += f(x);
    }
    
    return sum * delta;
}

// Riemann sum using right endpoint method
double riemann_sum_right(double a, double b, long n) {
    double delta = (b - a) / n;
    double sum = 0.0;
    
    #pragma omp parallel for reduction(+:sum)
    for (long i = 1; i <= n; i++) {
        double x = a + i * delta;
        sum += f(x);
    }
    
    return sum * delta;
}

// Riemann sum using midpoint method
double riemann_sum_midpoint(double a, double b, long n) {
    double delta = (b - a) / n;
    double sum = 0.0;
    
    #pragma omp parallel for reduction(+:sum)
    for (long i = 0; i < n; i++) {
        double x = a + (i + 0.5) * delta;
        sum += f(x);
    }
    
    return sum * delta;
}

int main(int argc, char *argv[]) {
    double a = 0.0;      // Lower bound
    double b = 1.0;      // Upper bound
    long n = 10000000;   // Number of rectangles
    
    // Set number of threads (optional)
    omp_set_num_threads(8);
    
    printf("Computing Riemann Sum for f(x) = x^2 from %.2f to %.2f\n", a, b);
    printf("Number of rectangles: %ld\n", n);
    printf("Number of threads: %d\n\n", omp_get_max_threads());
    
    // Left endpoint method
    double start = omp_get_wtime();
    double result_left = riemann_sum_left(a, b, n);
    double end = omp_get_wtime();
    printf("Left Riemann Sum: %.10f (Time: %.6f seconds)\n", 
           result_left, end - start);
    
    // Right endpoint method
    start = omp_get_wtime();
    double result_right = riemann_sum_right(a, b, n);
    end = omp_get_wtime();
    printf("Right Riemann Sum: %.10f (Time: %.6f seconds)\n", 
           result_right, end - start);
    
    // Midpoint method
    start = omp_get_wtime();
    double result_mid = riemann_sum_midpoint(a, b, n);
    end = omp_get_wtime();
    printf("Midpoint Riemann Sum: %.10f (Time: %.6f seconds)\n", 
           result_mid, end - start);
    
    // Exact integral for f(x) = x^2 from 0 to 1 is 1/3
    printf("\nExact value (for x^2 from 0 to 1): %.10f\n", 1.0/3.0);
    
    return 0;
}