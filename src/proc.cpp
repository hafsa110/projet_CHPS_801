#include <omp.h>
#include <iostream>

int main() {
    int num_threads = omp_get_num_procs();
    omp_set_num_threads(num_threads);

    // Your parallel code goes here
    #pragma omp parallel
    printf("hey");

    std::cout << "Number of available cores: " << num_threads << std::endl;
    return 0;
}