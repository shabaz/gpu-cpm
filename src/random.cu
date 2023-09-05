//#include "random.h"

//based on "A Fast High Quality Pseudo Random Number Generator for nVidia CUDA"
//by W.B. Langdon
__device__ __forceinline__ int nextRandom(int seed) {
    const double a = 16807;
    const double m = 2147483647;

    const double reciprocal_m = 1.0/m;
    double temp = seed * a;
    return temp - m * floor(temp * reciprocal_m);
}

__device__ __forceinline__ double scaleRandomToUnity(int seed) {
    return double(seed) / 2147483647;
}

__device__ int scaleRandomToDelta(int seed) {
    return (seed % 3) - 1;
}

__device__ unsigned int getNeighbor(unsigned int x, int& seed, int dimension) {
    seed = nextRandom(seed);
    int dx = scaleRandomToDelta(seed);
    return (x + dx) & (dimension - 1);
    //return (x + dx + dimension) % dimension;

}
