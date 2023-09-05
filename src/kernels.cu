#include <cooperative_groups.h>
#include <stdio.h>

#include "kernels.h"
#include "random.cu"

__host__ __device__ unsigned int interleaveWithZeros(unsigned short input)  {
    unsigned int word = input;
    word = (word ^ (word << 8 )) & 0x00ff00ff00ff00ff;
    word = (word ^ (word << 4 )) & 0x0f0f0f0f0f0f0f0f;
    word = (word ^ (word << 2 )) & 0x3333333333333333;
    word = (word ^ (word << 1 )) & 0x5555555555555555;
    return word;
}





__host__ __device__ int calculateIndex(int x, int y, int dimension) {

    //unsigned int a = interleaveWithZeros(x);
    //unsigned int b = interleaveWithZeros(y);
    //return a | (b << 1);

    //const int threadx = x / 4;
    //const int thready = y / 4;
    //const int inx = x%4;
    //const int iny = y%4;
    return y * dimension + x;
    //const int stride = 32;
    //return y * dimension + x/stride + x%stride * dimension/stride;
    //return (index/32) + (index%32) * dimension * dimension / 32;
    //const int stride = (dimension/4) * (dimension/4);
    //return (y/4) * (dimension/4) + (x/4) + ((dimension/4) * (dimension/4)) * ((y%4)*4 + (x%4));


}

__host__ __device__ int calculateIndex(int x, int y, int z, int dimension) {
    return z * dimension * dimension + y * dimension + x;
}

__device__ int spin(unsigned int cellId) {
    return cellId & (256*256*256-1);
}

__constant__ __device__ Settings constantSettings;

__host__ void setSettings(Settings& settings) {
    cudaMemcpyToSymbol(constantSettings, &settings, sizeof(Settings), 0, cudaMemcpyHostToDevice);
    
}

__global__
void resetAreasAndCircumferences(unsigned int* areas, unsigned int* circumferences) {
    int x = threadIdx.x;
    areas[x] = 0;
    circumferences[x] = 0;
}


__device__ void subtractFromCentroid(unsigned long long int* centroid, int x, int dimension) {
    while (true) {
        unsigned long long int old_x = *centroid;
        unsigned long long int x_centroid = *centroid;

        int* vals =(int*) &x_centroid;
        int& count = vals[0];
        int& sum = vals[1];

        int offset = sum - (x * count);
        offset /= dimension/2 * count;
        offset *= dimension;

        count--;
        sum -= x + offset;
        if (sum > count * dimension) {
            sum -= count * dimension;
        }
        if (sum < 0) {
            sum += count * dimension;
        }

        auto valAfterCAS = atomicCAS(centroid, old_x, x_centroid);

        if (valAfterCAS == old_x) 
            return;
    }
}

__device__ void addToCentroid(unsigned long long int* centroid, int x, int dimension) {
    while (true) {
        unsigned long long int old_x = *centroid;
        unsigned long long int x_centroid = *centroid;

        int* vals =(int*) &x_centroid;
        int& count = vals[0];
        int& sum = vals[1];

        int offset = sum - (x * count);
        offset /= dimension/2 * count;
        offset *= dimension;

        count++;
        sum += x + offset;
        if (sum > count * dimension) {
            sum -= count * dimension;
        }
        if (sum < 0) {
            sum += count * dimension;
        }

        auto valAfterCAS = atomicCAS(centroid, old_x, x_centroid);

        if (valAfterCAS == old_x) 
            return;
    }
}



__device__ int lock(unsigned int firstId, int firstArea, int firstDelta, 
        unsigned int secondId, int secondArea, int secondDelta, unsigned int* writeAreas) {
    if (firstId != 0) {
        unsigned int old = atomicCAS(writeAreas + firstId, firstArea, firstArea+firstDelta);
        if (old != firstArea) {
            return 0;
        }
    }
    if (secondId != 0) {
        unsigned int old = atomicCAS(writeAreas + secondId, secondArea, secondArea+secondDelta);
        if (old != secondArea) {
            if (firstId != 0) {
                writeAreas[firstId] = firstArea;
            }
            return 0;
        }
    }
    return 1;
}

__device__ double getCoordinate(unsigned long long int centroid) {
    int* vals =(int*) &centroid;
    int& count = vals[0];
    int& sum = vals[1];
    double x = (double)sum/(double)count;
    return x;
}

__device__ double wrap(double x, int dimension) {
    if (x > dimension/2)
        x -= dimension;
    if (x < -dimension/2)
        x += dimension;
    return x;
}

__global__ void updatePrefDirKernel(double* prefDirs, int* types, unsigned long long int* centroids, unsigned long long int* prevCentroids, int activeCells, int dimension) {

    int cellId = threadIdx.x + blockIdx.x * 1024;
    int type = types[cellId];
    double persistence = constantSettings.types[type].persistenceDiffusion;

    if (cellId > 0 && cellId < activeCells+1) {
        double prefX = prefDirs[cellId*3 + 0];
        double prefY = prefDirs[cellId*3 + 1];
        double prefZ = prefDirs[cellId*3 + 2];

        double currentX = getCoordinate(centroids[cellId*3 + 0]);
        double currentY = getCoordinate(centroids[cellId*3 + 1]);
        double currentZ = getCoordinate(centroids[cellId*3 + 2]);

        double prevX = getCoordinate(prevCentroids[cellId*3 + 0]);
        double prevY = getCoordinate(prevCentroids[cellId*3 + 1]);
        double prevZ = getCoordinate(prevCentroids[cellId*3 + 2]);



        double dirX = currentX - prevX;
        double dirY = currentY - prevY;
        double dirZ = currentZ - prevZ;

        double length = sqrt(dirX * dirX + dirY * dirY + dirZ * dirZ);
        if (length == 0)
            return;

        dirX = wrap(dirX, dimension);
        dirY = wrap(dirY, dimension);
        dirZ = wrap(dirZ, dimension);

        dirX /= length;
        dirY /= length;
        dirZ /= length;



        double newPrefX = dirX * (1-persistence) + prefX * persistence;
        double newPrefY = dirY * (1-persistence) + prefY * persistence;
        double newPrefZ = dirZ * (1-persistence) + prefZ * persistence;
        length = sqrt(newPrefX * newPrefX + newPrefY * newPrefY + newPrefZ * newPrefZ);

        newPrefX /= length;
        newPrefY /= length;
        newPrefZ /= length;

        
        prefDirs[cellId*3 + 0] = newPrefX;
        prefDirs[cellId*3 + 1] = newPrefY;
        prefDirs[cellId*3 + 2] = newPrefZ;
    }
}


#include "kernels2d.cu"
#include "kernels3d.cu"
