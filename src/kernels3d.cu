

__global__
void calculateAreasAndCircumferences3d(unsigned int* cellIds, unsigned int* areas, 
        unsigned int* writeAreas, unsigned int* circumferences, 
        unsigned long long int* centroids, int dimension) {

    int x = blockIdx.x;
    int y = blockIdx.y;
    int z = threadIdx.x + blockIdx.z * 1024;

    int index = calculateIndex(x, y, z, dimension);
    int cellId = spin(cellIds[index]);
    int type = cellIds[index] >> 24;

    if (cellId > 0) {
        atomicAdd(areas + cellId, 1);
        atomicAdd(writeAreas + cellId, 1);

        if (!constantSettings.types[type].fixed) {
            addToCentroid(centroids + cellId * 3, x, dimension);
            addToCentroid(centroids + cellId * 3 + 1, y, dimension);
            addToCentroid(centroids + cellId * 3 + 2, z, dimension);
        }
    }

    unsigned char interfaceCount = 0;
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dz = -1; dz <= 1; dz++) {
                int nx = (x + dx + dimension) % dimension;
                int ny = (y + dy + dimension) % dimension;
                int nz = (z + dz + dimension) % dimension;

                int nIndex = calculateIndex(nx, ny, nz, dimension);
                int nCellId = spin(cellIds[nIndex]);
                if (nCellId != cellId) {
                    interfaceCount++;
                }
            }
        }
    }

    if (cellId>0) {
        atomicAdd(circumferences + spin(cellId), interfaceCount);
    }
}



template<bool sharedToggle, bool actToggle, bool perimeterToggle, 
    bool persistenceToggle, bool chemotaxisToggle, bool fixedToggle,
    bool centroidToggle>
__global__
void copyAttempt3d(unsigned int* cellIds, unsigned int* areas, unsigned int* writeAreas,
        unsigned int* circumferences, unsigned long long* centroids,
        int dimension, int seed, unsigned int* acts, int tick, 
        unsigned int* analytics, int iterations, 
        int positionsPerThread, int positionsPerCheckerboard, 
        int updatesPerCheckerboardSwitch, int updatesPerBarrier,
        bool blockSync, bool globalSync, bool cellSync, bool insideThreadLoop,
        double* field, double* preferredDirection, const int sharedDim,
        
        bool globalCheckerboard, int globalCheckerboardX, int globalCheckerboardY, int globalCheckerboardZ) {
    int basex = (blockIdx.x * blockDim.x + threadIdx.x)*positionsPerThread;
    int basey = (blockIdx.y * blockDim.y + threadIdx.y)*positionsPerThread;
    int basez = (blockIdx.z * blockDim.z + threadIdx.z)*positionsPerThread;

    cooperative_groups::grid_group grp = cooperative_groups::this_grid();
    extern __shared__ unsigned int s[];

    int global_blocks = 1;
    if constexpr(sharedToggle) {
        global_blocks = positionsPerThread/(positionsPerCheckerboard*2);
        positionsPerThread = positionsPerCheckerboard*2;
    }

    int checkerBoards = positionsPerThread / positionsPerCheckerboard;

    int checkerboardMultiplier = (positionsPerCheckerboard * positionsPerCheckerboard * positionsPerCheckerboard)/updatesPerCheckerboardSwitch;


    int barrierCounter = -1;


    seed = nextRandom(calculateIndex(basex, basey, basez, dimension)+seed);
    tick--;

    for (int iteration = 0; iteration < iterations; iteration++) {
        tick++;
        for (int outer = 0; outer < checkerboardMultiplier; outer++) {
        for (int blockx = 0; blockx < global_blocks; blockx++) {
        for (int blocky = 0; blocky < global_blocks; blocky++) {
        for (int blockz = 0; blockz < global_blocks; blockz++) {

    if constexpr(sharedToggle) {
    __syncthreads(); 
    //basex = (blockIdx.x * blockDim.x + threadIdx.x)*positionsPerThread 
   //   + blockx * blockDim.x * gridDim.x * positionsPerThread;
  //basey = (blockIdx.y * blockDim.y + threadIdx.y)*positionsPerThread
 //     + blocky * blockDim.y * gridDim.y * positionsPerThread;

    basex = blockIdx.x * global_blocks * blockDim.x * positionsPerThread +
        threadIdx.x * positionsPerThread + blockx * blockDim.x * positionsPerThread;
    basey = blockIdx.y * global_blocks * blockDim.y * positionsPerThread +
        threadIdx.y * positionsPerThread + blocky * blockDim.y * positionsPerThread;
    basez = blockIdx.z * global_blocks * blockDim.z * positionsPerThread +
        threadIdx.z * positionsPerThread + blockz * blockDim.z * positionsPerThread;


    for(int k = 0; k < positionsPerThread; k++) {
        for(int j = 0; j < positionsPerThread; j++) {
            for(int i = 0; i < positionsPerThread; i++) {
                // coordinates within block
                //int localX = i + threadIdx.x * positionsPerThread;
                int localX = threadIdx.x  + blockDim.x * i;
                int localY = j + threadIdx.y * positionsPerThread;
                int localZ = k + threadIdx.z * positionsPerThread;

                int globalX = blockIdx.x * global_blocks * blockDim.x * positionsPerThread +
                    threadIdx.x + blockDim.x * i + 
                    blockx * blockDim.x * positionsPerThread;
                //int globalX = (blockIdx.x * blockDim.x)*positionsPerThread + i * blockDim.x + threadIdx.x
                //+ blockx * blockDim.x * gridDim.x * positionsPerThread;
                //int globalX =  blockDim.x + threadIdx.x;
                //int globalX = i + basex;
                int globalY = j + basey;
                int globalZ = k + basez;

                int localIndex = localZ * sharedDim * sharedDim + localY * sharedDim + localX;
                int index = calculateIndex(globalX, globalY, globalZ, dimension);
                s[localIndex] = cellIds[index];
            }
        }
    }
    grp.sync();
    }


            for(int checkerboardX = globalCheckerboardX; checkerboardX < checkerBoards; checkerboardX++) {
                for(int checkerboardY = globalCheckerboardY; checkerboardY < checkerBoards; checkerboardY++) {
                    for(int checkerboardZ = globalCheckerboardZ; checkerboardZ < checkerBoards; checkerboardZ++) {
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


                            int nx, ny, nz, x, y, z, index, nIndex;

                            seed = nextRandom(seed);
                            int a = seed % (positionsPerCheckerboard);
                            seed = nextRandom(seed);
                            int b = seed % (positionsPerCheckerboard);
                            seed = nextRandom(seed);
                            int c = seed % (positionsPerCheckerboard);

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
                            nz = basez + checkerboardZ * positionsPerCheckerboard + c;
                            nIndex = calculateIndex(nx, ny, nz, dimension);


                            x = getNeighbor(nx, seed, dimension);
                            y = getNeighbor(ny, seed, dimension);
                            z = getNeighbor(nz, seed, dimension);

                            // these while loops make behaviour of 
                            // one MCS match with serial CPM
                            while (x == nx && y == ny && z == nz) {
                                //randomly pick update direction
                                x = getNeighbor(nx, seed, dimension);
                                y = getNeighbor(ny, seed, dimension);
                                z = getNeighbor(nz, seed, dimension);
                            }
                            index = calculateIndex(x, y, z, dimension);
                            } else {
                            x = basex + checkerboardX * positionsPerCheckerboard + a;
                            y = basey + checkerboardY * positionsPerCheckerboard + b;
                            z = basez + checkerboardZ * positionsPerCheckerboard + c;
                            index = calculateIndex(x, y, z, dimension);

                            //randomly pick update direction
                            nx = getNeighbor(x, seed, dimension);
                            ny = getNeighbor(y, seed, dimension);
                            nz = getNeighbor(z, seed, dimension);

                            // these while loops make behaviour of 
                            // one MCS match with serial CPM
                            while (nx == x && ny == y && nz == y) {
                                nx = getNeighbor(x, seed, dimension);
                                ny = getNeighbor(y, seed, dimension);
                                nz = getNeighbor(z, seed, dimension);
                            }
                            nIndex = calculateIndex(nx, ny, nz, dimension);
                            }



                            int localnx, localny, localnz, localx, localy, localz, localindex, localnindex;
                            int cellId, nCellId;


                            if constexpr (sharedToggle) {
                                localnx = threadIdx.x * positionsPerThread + checkerboardX * positionsPerCheckerboard + a;
                                localny = threadIdx.y * positionsPerThread + checkerboardY * positionsPerCheckerboard + b;
                                localnz = threadIdx.z * positionsPerThread + checkerboardZ * positionsPerCheckerboard + c;
                                localnindex = localnz * sharedDim * sharedDim + localny * sharedDim + localnx;

                                localx = localnx + x - nx;
                                localy = localny + y - ny;
                                localz = localnz + z - nz;
                                localindex = localz * sharedDim * sharedDim + localy * sharedDim + localx;

                                nCellId = s[localnindex];
                                if (localx < 0 || localy < 0 || localz <0 || localx > sharedDim - 1 || localy > sharedDim - 1 || localz > sharedDim - 1) {
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

                                float dx, dy, dz;

                                if constexpr(persistenceToggle || chemotaxisToggle) {
                                dx = nx - x;
                                dy = ny - y;
                                dz = nz - z;

                                if (dx > dimension/2) 
                                    dx -= dimension;
                                if (dx < -dimension/2) 
                                    dx += dimension;
                                if (dy > dimension/2) 
                                    dy -= dimension;
                                if (dy < -dimension/2) 
                                    dy += dimension;
                                if (dz > dimension/2) 
                                    dz -= dimension;
                                if (dz < -dimension/2) 
                                    dz += dimension;

                                float dlength = sqrt(dx * dx + dy * dy + dz * dz);
                                dx /= dlength;
                                dy /= dlength;
                                dz /= dlength;
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


                                if constexpr(chemotaxisToggle) {
                                // chemotaxis constraint 
                                double chemoLambda = constantSettings.types[newType].chemotaxis;
                                if (newType == 0)
                                    chemoLambda = constantSettings.types[oldType].chemotaxis;
                                if (chemoLambda != 0) {
                                    double fieldx = field[index];
                                    double fieldy = field[index + dimension * dimension * dimension];
                                    double fieldz = field[index + 2 * dimension * dimension * dimension];
                                    energy += -chemoLambda * (dx * fieldx + dy * fieldy + dz * fieldz);
                                }
                                }

                                if constexpr(persistenceToggle) {
                                //preferential direction constraint
                                double prefDirX = preferredDirection[sSpin * 3 + 0];
                                double prefDirY = preferredDirection[sSpin * 3 + 1];
                                double prefDirZ = preferredDirection[sSpin * 3 + 2];

                                double persistenceLambda = constantSettings.types[newType].persistenceLambda;
                                if (newType == 0)
                                    persistenceLambda = constantSettings.types[oldType].persistenceLambda;

                                if (sSpin == 0) {
                                    prefDirX = preferredDirection[nSpin * 3 + 0];
                                    prefDirY = preferredDirection[nSpin * 3 + 1];
                                    prefDirZ = preferredDirection[nSpin * 3 + 2];
                                }

                                energy += -persistenceLambda * (dx * prefDirX + dy * prefDirY + dz * prefDirZ);
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
                                for (int k = -1; k <= 1; k++) {
                                    for (int j = -1; j <= 1; j++) {
                                        for (int i = -1; i <= 1; i++) {
                                            if (i == 0 && j == 0 && k == 0) continue;
                                            int nnx = (nx + i + dimension) % dimension;
                                            int nny = (ny + j + dimension) % dimension;
                                            int nnz = (nz + k + dimension) % dimension;

                                            int nnIndex = calculateIndex(nnx, nny, nnz, dimension);

                                            unsigned int nnCellId;


                                            if constexpr(sharedToggle) {

                                            int localnnx = localnx + i;
                                            int localnny = localny + j;
                                            int localnnz = localnz + k;
                                            int localnnIndex = localnnz * sharedDim * sharedDim + localnny * sharedDim + localnnx;


                                            if (localnnx < 0 || localnny < 0 || localnnz < 0|| localnnx > sharedDim - 1 || localnny > sharedDim - 1 || localnnz > sharedDim - 1) {
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
                                        for (int k = -1; k <= 1; k++) {
                                            if (i == 0 && j == 0 && k == 0) continue;
                                            int nnx = (x + i + dimension) % dimension;
                                            int nny = (y + j + dimension) % dimension;
                                            int nnz = (z + k + dimension) % dimension;

                                            int nnIndex = calculateIndex(nnx, nny, nnz, dimension);

                                            unsigned int nnAct;
                                            unsigned int nnCellId;

                                            if constexpr (sharedToggle) {
                                            int localnnx = localx + i;
                                            int localnny = localy + j;
                                            int localnnz = localz + k;
                                            int localnnIndex = localnnz * sharedDim * sharedDim + localnny * sharedDim + localnnx;
                                            if (localnnx < 0 || localnny < 0 || localnnz < 0 || localnnx > sharedDim - 1 || localnny > sharedDim - 1 || localnnz > sharedDim - 1) {
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
                                }
                                if (cellId != 0) {
                                    sourceAct = __powf(sourceAct, 1.0/count);
                                } else {
                                    sourceAct = 0;
                                }


                                // act constraint 

                                if (lambdaAct > 0) {
                                    energy += -lambdaAct/maxAct * (sourceAct - targetAct);
                                }
                                }
                                }


                                // adhession constraint 
                                energy += newNeighborEnergy - neighborEnergy;




                                if constexpr (perimeterToggle) {
                                // perimeter constraint
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
                                    if constexpr(sharedToggle) {
                                        s[localnindex] = cellId;
                                    }
                                    cellIds[nIndex] = cellId;

                                    if constexpr(actToggle) {
                                    acts[nIndex] = tick;
                                    }

                                    if (cellId != 0) {
                                        atomicAdd(areas + sSpin, 1);
                                        if constexpr(persistenceToggle || centroidToggle) {
                                            addToCentroid(centroids + sSpin * 3, nx, dimension);
                                            addToCentroid(centroids + sSpin * 3+1, ny, dimension);
                                            addToCentroid(centroids + sSpin * 3+2, nz, dimension);
                                        }
                                        if constexpr(perimeterToggle)
                                            atomicAdd(circumferences + sSpin, newPerimeterSource - oldPerimeterSource);
                                    }
                                    if (nCellId != 0) {
                                        atomicSub(areas + nSpin, 1);
                                        if constexpr(persistenceToggle || centroidToggle) {
                                            subtractFromCentroid(centroids + nSpin * 3, nx, dimension);
                                            subtractFromCentroid(centroids + nSpin * 3+1, ny, dimension);
                                            subtractFromCentroid(centroids + nSpin * 3+2, nz, dimension);
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
    }
}



template <std::size_t...Is>
auto call_f_helper3d(int i, std::index_sequence<Is...>)
{
    //hsing f_t = void();
    using f_t = void (unsigned int*, unsigned int*, unsigned int*,
        unsigned int*, unsigned long long*, int, int, unsigned int*, int, 
        unsigned int*, int, int, int, int, int, bool, bool, bool, bool,
        double*, double*, int, bool, int, int, int);
    f_t* fs[] = {&copyAttempt3d<(Is >> 0) & 1, (Is >> 1) & 1, (Is >> 2) & 1, (Is >> 3) & 1, (Is >> 4) & 1, (Is >> 5) & 1, (Is >> 6) & 1>...};
    return fs[i];

    //fs[i]();
}

// The runtime dispather
auto call_f3d(bool b1, bool b2, bool b3, bool b4, bool b5, bool b6, bool b7)
{
    return call_f_helper3d(b1 << 0 | b2 << 1 | b3 << 2 | b4 << 3 | b5 << 4 | b6 << 5 | b7 << 6,std::make_index_sequence<128>());  
}



