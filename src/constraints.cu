//#include "constraints.h"
//#include "lattice.h"
//#include "settings.h"
//#include <stdio.h>

const int TEMPERATURE = 10;
//const int LAMBDA_AREA = 50;
//const int TARGET_AREA = 500;

const int LAMBDA_PERIMETER = 1;
const int TARGET_PERIMETER = 850;

const double LAMBDA_ACT = 1000;
const double ACT_MAX = 40;

__device__ double getNeighbourEnergy(unsigned int cellId, int x, int y, 
        int dimension, unsigned int* cellIds, Settings& settings, int blockX, 
        int blockY, int positionsShared, unsigned int* shared) {
    double energy = 0;
    int type = cellId >> 24;
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            int nx = (x + i + dimension) % dimension;
            int ny = (y + j + dimension) % dimension;

            int nIndex = calculateIndex(nx, ny, dimension);
            int nSharedIndex = calculateSharedIndex(nx, ny, blockX, blockY, positionsShared);

            unsigned int nCellId;
            if (nSharedIndex != -1) {
                nCellId = shared[nSharedIndex];
            } else {
                nCellId = cellIds[nIndex];
            }
            int nType = nCellId >> 24;
            energy += settings.types[type].adhesion[nType];
        }
    }
    return energy;
}

__device__ void updatePerimeterDelta(int sourceId, int targetId, int x, int y, 
        int dimension, unsigned int* cellIds, unsigned int* perimeters) {
    int oldPerimeterSource = 0;
    int newPerimeterSource = 0;

    int oldPerimeterTarget = 0;
    int newPerimeterTarget = 0;

    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            if (i == 0 && j == 0) continue;
            int nx = (x + i + dimension) % dimension;
            int ny = (y + j + dimension) % dimension;

            int nIndex = calculateIndex(nx, ny, dimension);
            int nCellId = cellIds[nIndex];
            if (nCellId == sourceId)
                oldPerimeterSource++;
            if (nCellId != sourceId)
                newPerimeterSource++;
            if (nCellId != targetId)
                oldPerimeterTarget++;
            if (nCellId == targetId)
                newPerimeterTarget++;
        }
    }

    if (sourceId != 0) {
        atomicAdd(perimeters + spin(sourceId), newPerimeterSource - oldPerimeterSource);
    }

    if (targetId != 0) {
        atomicAdd(perimeters + spin(targetId), newPerimeterTarget - oldPerimeterTarget);
    }
}

__device__ double getAreaDelta(unsigned int cellId, unsigned int nCellId, 
        unsigned int* areas, Settings& settings) {
    double energyDelta = 0; 
    if (cellId != 0) {
        int area = areas[spin(cellId)];
        int type = cellId >> 24;
        int lambda = settings.types[type].areaLambda;
        int target = settings.types[type].areaTarget;
        double areaEnergy = lambda * (area - target)*(area - target);
        double newAreaEnergy = lambda * (area + 1 - target)*(area + 1 - target);
        energyDelta += newAreaEnergy - areaEnergy;
        //printf("id %d area %d target %d delta %f", spin(cellId), area, target, newAreaEnergy - areaEnergy);
    }

    if (nCellId != 0) {
        int area = areas[spin(nCellId)];
        int type = nCellId >> 24;
        int lambda = settings.types[type].areaLambda;
        int target = settings.types[type].areaTarget;
        double areaEnergy = lambda * (area - target)*(area - target);
        double newAreaEnergy = lambda * (area - 1 - target)*(area - 1 - target);
        energyDelta += newAreaEnergy - areaEnergy;
        //printf("id %d area %d target %d delta %f", spin(cellId), area, target, newAreaEnergy - areaEnergy);
    }
    return energyDelta;
}

__device__ double getPerimeterDelta(int sourceId, int targetId, int x, int y, 
        int dimension, unsigned int* cellIds, unsigned int* perimeters) {
    double delta = 0;

    int oldPerimeterSource = 0;
    int newPerimeterSource = 0;

    int oldPerimeterTarget = 0;
    int newPerimeterTarget = 0;

    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            if (i == 0 && j == 0) continue;
            int nx = (x + i + dimension) % dimension;
            int ny = (y + j + dimension) % dimension;

            int nIndex = calculateIndex(nx, ny, dimension);
            int nCellId = cellIds[nIndex];
            if (nCellId == sourceId)
                oldPerimeterSource++;
            if (nCellId != sourceId)
                newPerimeterSource++;
            if (nCellId != targetId)
                oldPerimeterTarget++;
            if (nCellId == targetId)
                newPerimeterTarget++;
        }
    }


    if (sourceId != 0) {
        int currentPerimeter = perimeters[spin(sourceId)];
        int newPerimeter = currentPerimeter - oldPerimeterSource+
            newPerimeterSource;
        int targetPerimeter = TARGET_PERIMETER;
        int lambda = LAMBDA_PERIMETER;

        delta += 
            lambda *((newPerimeter - targetPerimeter)*(newPerimeter - targetPerimeter) - 
                    (currentPerimeter - targetPerimeter)*(currentPerimeter - targetPerimeter));
    }
    if (targetId != 0) {
        int currentPerimeter = perimeters[spin(targetId)];
        int newPerimeter = currentPerimeter - oldPerimeterTarget +
            newPerimeterTarget;
        int targetPerimeter = TARGET_PERIMETER;
        int lambda = LAMBDA_PERIMETER;
        delta += 
            lambda *((newPerimeter - targetPerimeter)*(newPerimeter - targetPerimeter) - 
                    (currentPerimeter - targetPerimeter)*(currentPerimeter - targetPerimeter));
    }

    return delta;

}

__device__ double getActSource(int sourceId, int targetId, int sourceIndex,
        int sourceX, int sourceY, int targetX, int targetY,
        int targetIndex, int dimension, unsigned int* cellIds, 
        unsigned int* acts, int tick) {
    const auto maxAct = ACT_MAX;
    double sourceAct = max(maxAct+acts[sourceIndex]-tick, 0.0);
    if (sourceId != 0) {
        int count = 1;


        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                if (i == 0 && j == 0) continue;
                int nx = (sourceX + i + dimension) % dimension;
                int ny = (sourceY + j + dimension) % dimension;

                int nIndex = calculateIndex(nx, ny, dimension);
                int neighborId = cellIds[nIndex];
                if (neighborId == sourceId) {
                    auto act = max(maxAct+acts[nIndex]-tick, 0.0);
                    sourceAct *= act;
                    count++;
                }
            }
        }

        sourceAct = __powf(sourceAct, 1.0/count);
    } else {
        sourceAct = 0;
    }


    double targetAct = max(maxAct+acts[targetIndex]-tick, 0.0);
    if (targetId != 0) {
        int count = 1;

        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                if (i == 0 && j == 0) continue;
                int nx = (targetX + i + dimension) % dimension;
                int ny = (targetY + j + dimension) % dimension;

                int nIndex = calculateIndex(nx, ny, dimension);
                int neighborId = cellIds[nIndex];
                if (neighborId == targetId) {
                    auto act = max(maxAct+acts[nIndex]-tick, 0.0);
                    targetAct *= act;
                    count++;
                }
            }
        }
        targetAct = __powf(targetAct, 1.0/count);
    } else {
        targetAct = 0;
    }


    return targetAct;
}
    


__device__ double getActDelta(int sourceId, int targetId, int sourceIndex,
        int sourceX, int sourceY, int targetX, int targetY,
        int targetIndex, int dimension, unsigned int* cellIds, 
        unsigned int* acts, int tick) {
    const auto maxAct = ACT_MAX;
    double sourceAct = max(maxAct+acts[sourceIndex]-tick, 0.0);
    if (sourceId != 0) {
        int count = 1;


        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                if (i == 0 && j == 0) continue;
                int nx = (sourceX + i + dimension) % dimension;
                int ny = (sourceY + j + dimension) % dimension;

                int nIndex = calculateIndex(nx, ny, dimension);
                int neighborId = cellIds[nIndex];
                if (neighborId == sourceId) {
                    auto act = max(maxAct+acts[nIndex]-tick, 0.0);
                    sourceAct *= act;
                    count++;
                }
            }
        }

        sourceAct = __powf(sourceAct, 1.0/count);
    } else {
        sourceAct = 0;
    }


    double targetAct = max(maxAct+acts[targetIndex]-tick, 0.0);
    if (targetId != 0) {
        int count = 1;

        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                if (i == 0 && j == 0) continue;
                int nx = (targetX + i + dimension) % dimension;
                int ny = (targetY + j + dimension) % dimension;

                int nIndex = calculateIndex(nx, ny, dimension);
                int neighborId = cellIds[nIndex];
                if (neighborId == targetId) {
                    auto act = max(maxAct+acts[nIndex]-tick, 0.0);
                    targetAct *= act;
                    count++;
                }
            }
        }
        targetAct = __powf(targetAct, 1.0/count);
    } else {
        targetAct = 0;
    }

    auto lambda = LAMBDA_ACT;
    if (maxAct == 0)
        return 0;
    else {
        return -lambda/maxAct * (sourceAct - targetAct);
    }



}

__device__ double getAdhesionEnergy(int cellId, int nCellId, int nx, int ny, 
        int dimension, unsigned int* cellIds, Settings& settings, int blockX, 
        int blockY, int positionsShared, unsigned int* shared) {
    double neighbourEnergy = getNeighbourEnergy(nCellId, nx, ny, dimension, cellIds, settings, blockX, blockY, positionsShared, shared);
    double newNeighbourEnergy = getNeighbourEnergy(cellId, nx, ny, dimension, cellIds, settings, blockX, blockY, positionsShared, shared);
    return newNeighbourEnergy - neighbourEnergy;
}

