#ifndef KERNELS_H_
#define KERNELS_H_

#include "settings.h"

__global__ void resetAreasAndCircumferences(unsigned int* areas, 
        unsigned int* circumferences);
__global__ void calculateAreasAndCircumferences(unsigned int* cellIds, unsigned int* areas, 
        unsigned int* writeAreas, unsigned int* circumferences, 
        unsigned long long int* centroids, int dimension);
template <bool sharedToggle, bool actToggle, bool perimeterToggle, 
         bool persistenceToggle, bool chemotaxisToggle, bool fixedToggle>
__global__ void copyAttempt(unsigned int* cellIds, unsigned int* areas, unsigned int* writeAreas,
        unsigned int* circumferences, unsigned long long* centroids,
        int dimension, int seed, unsigned int* acts, int tick, 
        unsigned int* analytics, int iterations, 
        int positionsPerThread, int positionsPerCheckerboard, 
        int updatesPerCheckerboardSwitch, int updatesPerBarrier,
        bool blockSync, bool globalSync, bool cellSync, bool insideThreadLoop,
        double* field, double* preferredDirection, const int sharedDim,
        bool globalCheckerboard, int globalCheckerboardX, int globalCheckerboardY);
__host__ void setSettings(Settings& settings);


__global__ void calculateAreasAndCircumferences3d(unsigned int* cellIds, unsigned int* areas, 
        unsigned int* writeAreas, unsigned int* circumferences, 
        unsigned long long int* centroids, int dimension);
template<bool sharedToggle, bool actToggle, bool perimeterToggle, 
    bool persistenceToggle, bool chemotaxisToggle, bool fixedToggle, bool centroidToggle>
__global__ void copyAttempt3d(unsigned int* cellIds, unsigned int* areas, unsigned int* writeAreas,
        unsigned int* circumferences, unsigned long long* centroids,
        int dimension, int seed, unsigned int* acts, int tick, 
        unsigned int* analytics, int iterations, 
        int positionsPerThread, int positionsPerCheckerboard, 
        int updatesPerCheckerboardSwitch, int updatesPerBarrier,
        bool blockSync, bool globalSync, bool cellSync, bool insideThreadLoop,
        double* field, double* preferredDirection, int sharedDim,
        bool globalCheckerboard, int globalCheckerboardX, int globalCheckerboardY, int globalCheckerboardZ);

__global__ void updatePrefDirKernel(double* prefDirs, int* types, unsigned long long int* centroids, unsigned long long int* prevCentroids, int activeCells, int dimension);


#endif // KERNELS_H_
