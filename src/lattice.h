#ifndef LATTICE_H_
#define LATTICE_H_

__host__ __device__ int calculateIndex(int x, int y, int dimension);
__host__ __device__ int calculateIndex(int x, int y, int z, int dimension);
__device__ int calculateSharedIndex(int x, int y, int blockX, int blockY, int positionsShared);
__device__ int spin(unsigned int cellId);

#endif // LATTICE_H_
