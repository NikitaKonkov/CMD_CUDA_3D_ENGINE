#include <stdio.h>
#include <cuda.h>
#include <vector>
#define N 256

__global__ void matrix(){


}

int main(int argc, char const *argv[])
{

    std::vector<uint_fast8_t> vec_0(N);

    int * out_0, target, source;

    cudaMalloc(&out_0,16*16*sizeof(int_fast8_t));

    cudaMemcpy(source, target,sizeof(uint8_t))

    matrix<<<1,32>>>();

    while(1){}
    return 0;
}
