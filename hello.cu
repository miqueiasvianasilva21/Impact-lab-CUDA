#include <stdio.h>


__global__ void helloFromGPU() {
    printf("Hello World from GPU thread %d!\n");
}

int main() {
    
    helloFromGPU<<<1, 1>>>();

    
    

    return 0;
}
