#ifndef CONSTRAINTS_H_
#define CONSTRAINTS_H_

struct Settings;

const int TEMPERATURE = 10;
//const int LAMBDA_AREA = 50;
//const int TARGET_AREA = 500;

const int LAMBDA_PERIMETER = 2;
const int TARGET_PERIMETER = 340;

const double LAMBDA_ACT = 140;
const double ACT_MAX = 40;

__device__ double getAreaDelta(unsigned int cellId, unsigned int nCellId, 
        unsigned int* areas, Settings& settings);
__device__ double getAdhesionEnergy(int cellId, int nCellId, int nx, int ny, 
        int dimension, unsigned int* cellIds, Settings& settings, int blockX,
        int blockY, int positionsShared, unsigned int* shared);

#endif // CONSTRAINTS_H_
