#include <utility> 
#include <cooperative_groups.h>
using namespace std;

__global__
void calculateAreasAndCircumferences(unsigned int* cellIds, unsigned int* areas, 
        unsigned int* writeAreas, unsigned int* circumferences, 
        unsigned long long int* centroids, int dimension) {

    int x = blockIdx.x;
    int y = threadIdx.x + blockIdx.y * 1024;

    int index = calculateIndex(x, y, dimension);
    int cellId = spin(cellIds[index]);
    if (cellId > 0) {
        atomicAdd(areas + cellId, 1);
        atomicAdd(writeAreas + cellId, 1);


        addToCentroid(centroids + cellId * 2, x, dimension);
        addToCentroid(centroids + cellId * 2 + 1, y, dimension);
    }

    unsigned char interfaceCount = 0;
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            int nx = (x + dx + dimension) % dimension;
            int ny = (y + dy + dimension) % dimension;

            int nIndex = calculateIndex(nx, ny, dimension);
            int nCellId = spin(cellIds[nIndex]);
            if (nCellId != cellId) {
                interfaceCount++;
            }
        }
    }

    if (cellId>0) {
        atomicAdd(circumferences + spin(cellId), interfaceCount);
    }
}


template<bool sharedToggle, bool actToggle, bool perimeterToggle, 
    bool persistenceToggle, bool chemotaxisToggle, bool fixedToggle>
__global__
void copyAttempt(unsigned int* cellIds, unsigned int* areas, unsigned int* writeAreas,
        unsigned int* circumferences, unsigned long long* centroids,
        int dimension, int seed, unsigned int* acts, int tick, 
        unsigned int* analytics, int iterations, 
        int positionsPerThread, int positionsPerCheckerboard, 
        int updatesPerCheckerboardSwitch, int updatesPerBarrier,
        bool blockSync, bool globalSync, bool cellSync, bool insideThreadLoop,
        double* field, double* preferredDirection, const int sharedDim, 
        bool globalCheckerboard, int globalCheckerboardX, int globalCheckerboardY) {

    extern __shared__ unsigned int s[];
    //__shared__ unsigned int s[64*64];



    cooperative_groups::grid_group grp = cooperative_groups::this_grid();



    int barrierCounter = -1;

    // blockidx * blockdim + threadidx gives us what thread # we are, along 
    // a certain dimension
    // multiplied with positionsPerThread gives us the base location for the 
    // area assigned to the current thread
    int basex = (blockIdx.x * blockDim.x + threadIdx.x)*positionsPerThread;
    int basey = (blockIdx.y * blockDim.y + threadIdx.y)*positionsPerThread;

    // global blocks is used when do outer layer of iterations when is 
    // shared memory mode (defaults to 1, i.e. no global block, when not in 
    // shared memory mode)
    int global_blocks = 1;
    if constexpr(sharedToggle) {
        // we take nr of positions originally asigned to this thread for a 
        // checkerboard sweep (positionsPerThread) and region the thread will 
        // cover in a checkerboard sweep in shared mem mode  (checkerboard area
        // times 2), and divide to get globabl blocks to compensate for
        // re-assigning positions per thread to 2x checkerboard section
        global_blocks = positionsPerThread/(positionsPerCheckerboard*2);
        positionsPerThread = positionsPerCheckerboard*2;
    }

    // how many checkerboard regions does each thread cycle through (always set
    // at 2 when in shared mem mode))
    int checkerBoards = positionsPerThread / positionsPerCheckerboard;

    // multiplier to compensate for switching inside checkerboard region more 
    // quickly than nr of positions for this region
    int checkerboardMultiplier = (positionsPerCheckerboard * positionsPerCheckerboard)/updatesPerCheckerboardSwitch;



    seed = nextRandom(calculateIndex(basex, basey, dimension)+seed);
    tick--;

    for (int iteration = 0; iteration < iterations; iteration++) {
        tick++;
        for (int outer = 0; outer < checkerboardMultiplier; outer++) { // compensate for switching from inner checkerboard region before all positions (statistically) got their turn
        for (int blockx = 0; blockx < global_blocks; blockx++) { // outer loop to compensate for smaller inner loop when in shared mem mode
        for (int blocky = 0; blocky < global_blocks; blocky++) {
            //grp.sync();

    if constexpr(sharedToggle) {
    __syncthreads(); 
    //basex = (blockIdx.x * blockDim.x + threadIdx.x)*positionsPerThread 
   //   + blockx * blockDim.x * gridDim.x * positionsPerThread;
  //basey = (blockIdx.y * blockDim.y + threadIdx.y)*positionsPerThread
 //     + blocky * blockDim.y * gridDim.y * positionsPerThread;

    // different indexing when we're in shared mem mode:
    // basex/y is base of current thread's active region, which is now also
    // determined by blockx/y loops
    // 
    basex = 
        // each threadblock has blockdim threads, and each thread has
        // global_blocks * positionsPerThread region assigned to it,
        // this line gives us offset of threadblock at certain blockidx
        blockIdx.x * global_blocks * blockDim.x * positionsPerThread +
        // this line gives offset of current thread relative to active region of
        // TB
        threadIdx.x * positionsPerThread + 
        // this gives offset of active region of threadblock with blockx/y
        // relative to full region assigned to current threadblock
        // each activeregion is sized # of threads in TB time positionsperthread
        // and blockx/y loops through these regions
        blockx * blockDim.x * positionsPerThread;
    basey = blockIdx.y * global_blocks * blockDim.y * positionsPerThread +
        threadIdx.y * positionsPerThread + blocky * blockDim.y * positionsPerThread;


    for(int j = 0; j < positionsPerThread; j++) {
        for(int i = 0; i < positionsPerThread; i++) {
            // coordinates within block
            //int localX = i + threadIdx.x * positionsPerThread;

            // during loading, indexing for current thread block is bit different
            // to allow coalesced reads

            // localX/Y are coordinates within the current active region of 
            // TB. We go through X axis sequentially based in threadidx.x so 
            // reads will coalesc. Each i iteration will then skip blockdim
            // positions, equal to number of threads that did a read in TB
            int localX = threadIdx.x  + blockDim.x * i;
            // for Y we have a positionsPerThread region assigned to current thread
            // and loop through it with j, skipping over the regions assigned 
            // to lower threads with threadIdx.y*positionsPerThread
            int localY = j + threadIdx.y * positionsPerThread;

            // pretty close to basex calculation, first line same
            int globalX = blockIdx.x * global_blocks * blockDim.x * positionsPerThread +
                //second line instead of giving offset of assigned region of thread
                //in active TB region, it uses the localX style sequential offsetting during loading
                threadIdx.x + blockDim.x * i + 
                //third line same as basex
                blockx * blockDim.x * positionsPerThread;

            //int globalX = (blockIdx.x * blockDim.x)*positionsPerThread + i * blockDim.x + threadIdx.x
        //+ blockx * blockDim.x * gridDim.x * positionsPerThread;
            //int globalX =  blockDim.x + threadIdx.x;
            //int globalX = i + basex;
            int globalY = j + basey;

            int localIndex = localY * sharedDim + localX;
            int index = calculateIndex(globalX, globalY, dimension);
            s[localIndex] = cellIds[index];
        }
    }
    // only sync with other thread blocks _after_ reads, so reads can overlap
    grp.sync();
    }
            // checkerboard looping, is at [assigned region per thread]/[checkerboard region]
            // or fixed at 2 when in shared memory mode
            for(int checkerboardX = globalCheckerboardX; checkerboardX < checkerBoards; checkerboardX++) {
                for(int checkerboardY = globalCheckerboardY; checkerboardY < checkerBoards; checkerboardY++) {
                    // multiple updates per checkerboard region
                    for (int inner = 0; inner < updatesPerCheckerboardSwitch; inner++) {

                        barrierCounter++;

                        if (barrierCounter == updatesPerBarrier) {
                            barrierCounter = 0;
                            if (globalCheckerboard) 
                                return;
                            if (blockSync)
                                __syncthreads();
                            if (globalSync) {
                                grp.sync();
                            }
                        }

                        int nx, ny, x, y, index, nIndex;

                        seed = nextRandom(seed);
                        int a = seed % (positionsPerCheckerboard);
                        seed = nextRandom(seed);
                        int b = seed % (positionsPerCheckerboard);

                        if constexpr (sharedToggle) {
                        // we flip picking a position and then neighbor
                        // if we're in shared mem mode, because it better interacts
                        // with the shared mem based caching
                        // we build up position by startng with region assigned to 
                        // thread (basex/y), offset  active checkerboard region
                        // with checkerboardX/Y*positionsPerCheckerboard 
                        // and then a/b can take on values inside current checkerboard region
                        nx = basex + checkerboardX * positionsPerCheckerboard + a;
                        ny = basey + checkerboardY * positionsPerCheckerboard + b;
                        nIndex = calculateIndex(nx, ny, dimension);

                        x = getNeighbor(nx, seed, dimension);
                        y = getNeighbor(ny, seed, dimension);

                        // these while loops make behaviour of 
                        // one MCS match with serial CPM
                        while (x == nx && y == ny) {
                            //randomly pick update direction
                            x = getNeighbor(nx, seed, dimension);
                            y = getNeighbor(ny, seed, dimension);
                        }
                        index = calculateIndex(x, y, dimension);
                        } else {
                        x = basex + checkerboardX * positionsPerCheckerboard + a;
                        y = basey + checkerboardY * positionsPerCheckerboard + b;
                        index = calculateIndex(x, y, dimension);

                        //randomly pick update direction
                        nx = getNeighbor(x, seed, dimension);
                        ny = getNeighbor(y, seed, dimension);

                        // these while loops make behaviour of 
                        // one MCS match with serial CPM
                        while (nx == x && ny == y) {
                            nx = getNeighbor(x, seed, dimension);
                            ny = getNeighbor(y, seed, dimension);
                        }
                        nIndex = calculateIndex(nx, ny, dimension);
                        }




                        int localnx, localny, localx, localy, localindex, localnindex;
                        int cellId, nCellId;
                        if constexpr (sharedToggle) {
                        // first term skips previous threads assigned regions in TB,
                        // second term skips previous checkerboard areas not active
                        // for current thread, and then a/b give position in current
                        // checkerboard region that got picked
                        localnx = threadIdx.x * positionsPerThread + checkerboardX * positionsPerCheckerboard + a;
                        localny = threadIdx.y * positionsPerThread + checkerboardY * positionsPerCheckerboard + b;
                        localnindex = localny * sharedDim + localnx;

                        localx = localnx + x - nx;
                        localy = localny + y - ny;
                        localindex = localy * sharedDim + localx;

                        nCellId = s[localnindex];
                        if (localx < 0 || localy < 0 || localx > sharedDim - 1 || localy > sharedDim - 1) {
                            cellId = cellIds[index];
                        } else {
                            cellId = s[localindex];
                        }
                        } else {
                            cellId = cellIds[index];
                            nCellId = cellIds[nIndex];
                        }



                        if(cellId != nCellId) {
                            int oldType = nCellId >> 24;
                            int newType = cellId >> 24;

                            // fixed (unmovable) cell type constraint
                            if constexpr(fixedToggle) {
                                if (constantSettings.types[oldType].fixed) {
                                    continue;
                                }

                                if (constantSettings.types[newType].fixed) {
                                    newType = 0;
                                    cellId = 0;
                                }
                            }

                            //atomicAdd(analytics, 1);

                            unsigned int sSpin = spin(cellId);
                            unsigned int nSpin = spin(nCellId);

                            int area = areas[sSpin];
                            int nArea = areas[nSpin];

                            float energy = 0;

                            float dx, dy;
                            if constexpr(persistenceToggle || chemotaxisToggle) {
                            dx = nx - x;
                            dy = ny - y;
                            float dlength = sqrt(dx * dx + dy * dy);
                            dx /= dlength;
                            dy /= dlength;
                            }

                            // act constraint 
                            double lambdaAct;
                            int maxAct;
                            if constexpr (actToggle) {
                            lambdaAct = constantSettings.types[oldType].actLambda;
                            maxAct = constantSettings.types[oldType].actMax;

                            if (oldType == 0) {
                                lambdaAct = constantSettings.types[newType].actLambda;
                                maxAct = constantSettings.types[newType].actMax;
                            } 
                            }

                            // chemotaxis constraint 
                            if constexpr (chemotaxisToggle) {
                            double chemoLambda = constantSettings.types[newType].chemotaxis;
                            if (newType == 0)
                                chemoLambda = constantSettings.types[oldType].chemotaxis;
                            if (chemoLambda != 0) {
                                double fieldx = field[index];
                                double fieldy = field[index + dimension * dimension];
                                energy += -chemoLambda * (dx * fieldx + dy * fieldy);
                            } 
                            }



                            double sourceAct, targetAct;
                            int count = 1;
                            if constexpr (actToggle) {
                            sourceAct = max((double)constantSettings.types[newType].actMax+acts[index]-tick, 0.0);
                            targetAct = max((double)constantSettings.types[oldType].actMax+acts[nIndex]-tick, 0.0);
                            }


                            int neighborEnergy = 0;
                            int newNeighborEnergy = 0;

                            int oldPerimeterSource = 0;
                            int newPerimeterSource = 0;

                            int oldPerimeterTarget = 0;
                            int newPerimeterTarget = 0;




                            //calculations for constraints that depend on target position of copy attempt
                            for (int j = -1; j <= 1; j++) {
                                for (int i = -1; i <= 1; i++) {
                                    if (i == 0 && j == 0) continue;
                                    int nnx = (nx + i + dimension) % dimension;
                                    int nny = (ny + j + dimension) % dimension;

                                    int nnIndex = calculateIndex(nnx, nny, dimension);
                                    unsigned int nnCellId;

                                    if constexpr(sharedToggle) {

                                    int localnnx = localnx + i;
                                    int localnny = localny + j;
                                    int localnnIndex = localnny * sharedDim + localnnx;


                                    if (localnnx < 0 || localnny < 0 || localnnx > sharedDim - 1 || localnny > sharedDim - 1) {
                                        nnCellId = cellIds[nnIndex];
                                    } else {
                                        nnCellId = s[localnnIndex];
                                    }
                                    } else {
                                        nnCellId = cellIds[nnIndex];
                                    }


                                    unsigned int nnAct;
                                    if constexpr (actToggle) {
                                    nnAct = acts[nnIndex];
                                    }
                                    int nnType = nnCellId >> 24;

                                    if constexpr (actToggle) {

                                    if (nCellId == nnCellId) {
                                        targetAct *= max((double)constantSettings.types[oldType].actMax+nnAct-tick, 0.0);
                                        count++;
                                    }
                                    }

                                    if (nCellId != nnCellId)
                                        neighborEnergy += constantSettings.types[oldType].adhesion[nnType] + constantSettings.types[oldType].stickiness[nnType];
                                    if (cellId != nnCellId)
                                        newNeighborEnergy += constantSettings.types[newType].adhesion[nnType];

                                    if constexpr(perimeterToggle) {
                                        if (nnCellId == cellId)
                                            oldPerimeterSource++;
                                        if (nnCellId != cellId)
                                            newPerimeterSource++;
                                        if (nnCellId != nCellId)
                                            oldPerimeterTarget++;
                                        if (nnCellId == nCellId)
                                            newPerimeterTarget++;
                                    }
                                }
                            }

                          if constexpr (actToggle)  {
                            if (nCellId != 0) {
                                targetAct = __powf(targetAct, 1.0/count);
                            } else {
                                targetAct = 0;
                            }
                          }




                            if constexpr (actToggle) {

                            if (lambdaAct !=0) {
                            //calculations for constraints that depend on source position of copy attempt
                            count = 1;
                            for (int i = -1; i <= 1; i++) {
                                for (int j = -1; j <= 1; j++) {
                                    if (i == 0 && j == 0) continue;
                                    int nnx = (x + i + dimension) % dimension;
                                    int nny = (y + j + dimension) % dimension;

                                    int nnIndex = calculateIndex(nnx, nny, dimension);

                                    unsigned int nnAct;
                                    unsigned int nnCellId;

                                    if constexpr (sharedToggle) {
                                    int localnnx = localx + i;
                                    int localnny = localy + j;
                                    int localnnIndex = localnny * sharedDim + localnnx;
                                    if (localnnx < 0 || localnny < 0 || localnnx > sharedDim - 1 || localnny > sharedDim - 1) {
                                        nnCellId = cellIds[nnIndex];
                                    } else {
                                        nnCellId = s[localnnIndex];
                                    }
                                    } else {
                                        nnCellId = cellIds[nnIndex];
                                    }

                                    nnAct = acts[nnIndex];

                                    if (cellId == nnCellId) {
                                        sourceAct *= max((double)constantSettings.types[newType].actMax+nnAct-tick, 0.0);
                                        count++;
                                    }
                                }
                            }
                            if (cellId != 0) {
                                sourceAct = __powf(sourceAct, 1.0/count);
                            } else {
                                sourceAct = 0;
                            }

                            energy += -lambdaAct/maxAct * (sourceAct - targetAct);
                            }
                            }


                            // adhession constraint 
                            energy += newNeighborEnergy - neighborEnergy;


                            if constexpr (perimeterToggle) {

                             //perimeter constraint
                            if (cellId != 0) {
                                int currentPerimeter = circumferences[spin(cellId)];
                                int newPerimeter = currentPerimeter - oldPerimeterSource+
                                    newPerimeterSource;

                                int targetPerimeter = constantSettings.types[newType].perimeterTarget;
                                double lambda = constantSettings.types[newType].perimeterLambda;

                                energy += 
                                    lambda *((newPerimeter - targetPerimeter)*(newPerimeter - targetPerimeter) - 
                                            (currentPerimeter - targetPerimeter)*(currentPerimeter - targetPerimeter));
                            }
                            if (nCellId != 0) {
                                int currentPerimeter = circumferences[spin(nCellId)];
                                int newPerimeter = currentPerimeter - oldPerimeterTarget +
                                    newPerimeterTarget;
                                int targetPerimeter = constantSettings.types[oldType].perimeterTarget;
                                double lambda = constantSettings.types[oldType].perimeterLambda;
                                energy += 
                                    lambda *((newPerimeter - targetPerimeter)*(newPerimeter - targetPerimeter) - 
                                            (currentPerimeter - targetPerimeter)*(currentPerimeter - targetPerimeter));
                            }
                            }


                            // area constraint
                            if (cellId != 0) {
                                auto lambda = constantSettings.types[newType].areaLambda;
                                auto target = constantSettings.types[newType].areaTarget;
                                energy += lambda * (1+2 * area - 2 * target);
                            }

                            if (nCellId != 0) {
                                auto lambda = constantSettings.types[oldType].areaLambda;
                                auto target = constantSettings.types[oldType].areaTarget;
                                energy += lambda * (1-2 * nArea + 2 * target);
                            }


                            seed = nextRandom(seed);
                            float r = scaleRandomToUnity(seed);

                            if (energy < 0 || r < expf(-energy/constantSettings.temperature)) {
                                //atomicAdd(analytics+1, 1);
                                if (cellSync)
                                    if (lock(sSpin, area, 1, nSpin, nArea, -1, writeAreas) == 0) {
                                        continue;
                                    }
                                //atomicAdd(analytics+2, 1);
                                if constexpr(actToggle) {
                                acts[nIndex] = tick;
                                }

                                //cellIds[nIndex] = cellId;
                                //s[localnindex] = cellId;

                                //if (localnx < 0 || localny < 0 || localnx > 63 || localny > 63) {
                                    //cellIds[nIndex] = cellId;
                                //} else {
                                    //s[localnindex] = cellId;
                                //}
                                if constexpr(sharedToggle) {
                                    s[localnindex] = cellId;
                                }
                                    //if (localnx == 0 || localny == 0 || localnx == 63 || localny == 63)
                                cellIds[nIndex] = cellId;



                                if (cellId != 0) {
                                    atomicAdd(areas + sSpin, 1);
                                    if constexpr(persistenceToggle) {
                                        addToCentroid(centroids + sSpin * 2, nx, dimension);
                                        addToCentroid(centroids + sSpin * 2+1, ny, dimension);
                                    }
                                    if constexpr(perimeterToggle)
                                        atomicAdd(circumferences + sSpin, newPerimeterSource - oldPerimeterSource);
                                }
                                if (nCellId != 0) {
                                    atomicSub(areas + nSpin, 1);
                                    if constexpr(persistenceToggle) {
                                        subtractFromCentroid(centroids + nSpin * 2, nx, dimension);
                                        subtractFromCentroid(centroids + nSpin * 2+1, ny, dimension);
                                    }
                                    if constexpr(perimeterToggle)
                                        atomicAdd(circumferences + nSpin, newPerimeterTarget - oldPerimeterTarget);
                                }
                            } 
                        }
                    }
                }
            }
        }
        }
        }

    }



}



template <std::size_t...Is>
auto call_f_helper(int i, std::index_sequence<Is...>)
{
    //hsing f_t = void();
    using f_t = void (unsigned int*, unsigned int*, unsigned int*,
        unsigned int*, unsigned long long*, int, int, unsigned int*, int, 
        unsigned int*, int, int, int, int, int, bool, bool, bool, bool,
        double*, double*, int, bool, int, int);
    f_t* fs[] = {&copyAttempt<(Is >> 0) & 1, (Is >> 1) & 1, (Is >> 2) & 1, (Is >> 3) & 1, (Is >> 4) & 1, (Is >> 5) & 1>...};
    return fs[i];

    //fs[i]();
}

// The runtime dispather
auto call_f(bool b1, bool b2, bool b3, bool b4, bool b5, bool b6)
{
    return call_f_helper(b1 << 0 | b2 << 1 | b3 << 2 | b4 << 3 | b5 << 4,std::make_index_sequence<64>());  
}


