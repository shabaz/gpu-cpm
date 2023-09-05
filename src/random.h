#ifndef RANDOM_H_
#define RANDOM_H_

__device__ int nextRandom(int seed);
__device__ double scaleRandomToUnity(int seed);
__device__ int scaleRandomToDelta(int seed);
__device__ int getNeighbor(int x, int& seed, int dimension);

#endif // RANDOM_H_

