#include <algorithm>
#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>
#include <random>
#include "cpm.h"
#include "lattice.h"
#include "kernels.h"

using namespace std;


template <std::size_t...Is>
auto kernel_table_2d(int i, std::index_sequence<Is...>)
{
    using f_t = void (unsigned int*, unsigned int*, unsigned int*,
        unsigned int*, unsigned long long*, int, int, unsigned int*, int, 
        unsigned int*, int, int, int, int, int, bool, bool, bool, bool,
        double*, double*, int, bool, int, int);
    f_t* fs[] = {&copyAttempt<(Is >> 0) & 1, (Is >> 1) & 1, (Is >> 2) & 1, (Is >> 3) & 1, (Is >> 4) & 1, (Is >> 5) & 1>...};
    return fs[i];
}

// The runtime dispather
auto get_2d_kernel_pointer(bool b1, bool b2, bool b3, bool b4, bool b5, bool b6)
{
    return kernel_table_2d(b1 << 0 | b2 << 1 | b3 << 2 | b4 << 3 | b5 << 4 | b6 << 5,std::make_index_sequence<64>());  
}

template <std::size_t...Is>
auto kernel_table_3d(int i, std::index_sequence<Is...>)
{
    using f_t = void (unsigned int*, unsigned int*, unsigned int*,
        unsigned int*, unsigned long long*, int, int, unsigned int*, int, 
        unsigned int*, int, int, int, int, int, bool, bool, bool, bool,
        double*, double*, int, bool, int, int, int);
    f_t* fs[] = {&copyAttempt3d<(Is >> 0) & 1, (Is >> 1) & 1, (Is >> 2) & 1, (Is >> 3) & 1, (Is >> 4) & 1, (Is >> 5) & 1, (Is >> 6) & 1>...};
    return fs[i];
}

// The runtime dispather
auto get_3d_kernel_pointer(bool b1, bool b2, bool b3, bool b4, bool b5, bool b6, bool b7)
{
    return kernel_table_3d(b1 << 0 | b2 << 1 | b3 << 2 | b4 << 3 | b5 << 4 | b6 << 5 | b7 << 6,std::make_index_sequence<128>());  
}





// Handle errors in CUDA:
void handleCudaError(cudaError_t cudaERR){
    if (cudaERR!=cudaSuccess){
        printf("CUDA ERROR : %s\n", cudaGetErrorString(cudaERR));
    }
}
void Cpm::setTemperature(int temperature){
    _settings.temperature = temperature;
    setSettings(_settings);
}


int Cpm::getDimensionality() {
    return _dimensionality;
}


vector<v3> Cpm::get3dCentroids() {
    vector<v3> centroids;
    cudaMemcpy(_centroids, _d_centroids, 
            3*(_nrOfCells+1)*sizeof(unsigned long long),
            cudaMemcpyDeviceToHost);
    for (int i = 0; i < _activeCells; i++) {
        int* vals =(int*) &_centroids[(i+1)*3];
        int& count = vals[0];
        int& sum = vals[1];
        double x = (double)sum/(double)count;

        vals =(int*) &_centroids[(i+1)*3+1];
        int& count2 = vals[0];
        int& sum2 = vals[1];
        double y = (double)sum2/(double)count2;
        
        vals =(int*) &_centroids[(i+1)*3+2];
        int& count3 = vals[0];
        int& sum3 = vals[1];
        double z = (double)sum3/(double)count3;
        centroids.push_back({x,y,z});
    }
    return centroids;
}

vector<v2> Cpm::get2dCentroids() {
    vector<v2> centroids;
    cudaMemcpy(_centroids, _d_centroids, 
            2*(_nrOfCells+1)*sizeof(unsigned long long),
            cudaMemcpyDeviceToHost);
    for (int i = 0; i < _activeCells; i++) {
        int* vals =(int*) &_centroids[(i+1)*2];
        int& count = vals[0];
        int& sum = vals[1];
        double x = (double)sum/(double)count;
        vals =(int*) &_centroids[(i+1)*2+1];
        int& count2 = vals[0];
        int& sum2 = vals[1];
        double y = (double)sum2/(double)count2;
        centroids.push_back({x,y});
    }
    return centroids;
}

void Cpm::setPersistence(int type, int time, double diffusion, double lambda) {
    _settings.types[type].persistenceTime = time;
    _settings.types[type].persistenceLambda = lambda;
    _settings.types[type].persistenceDiffusion = diffusion;
    setSettings(_settings);
}

void Cpm::setPerimeter(int type, int target, double lambda){
    _settings.types[type].perimeterLambda = lambda;
    _settings.types[type].perimeterTarget = target;
    setSettings(_settings);
}

void Cpm::setArea(int type, int target, double lambda){
    _settings.types[type].areaLambda = lambda;
    _settings.types[type].areaTarget = target;
    setSettings(_settings);
}

void Cpm::setAct(int type, int maxAct, double lambda){
    _settings.types[type].actLambda = lambda;
    _settings.types[type].actMax = maxAct;
    setSettings(_settings);
}

void Cpm::setAdhesion(int type, int otherType, int adhesion, int setBoth){
    _settings.types[type].adhesion[otherType] = adhesion;
    if (setBoth) {
        _settings.types[otherType].adhesion[type] = adhesion;
    }
    setSettings(_settings);
}

void Cpm::setStickiness(int type, int otherType, int stickiness){
    _settings.types[type].stickiness[otherType] = stickiness;
    setSettings(_settings);
}

void Cpm::setFixed(int type, int fixed){
    _settings.types[type].fixed = fixed;
    setSettings(_settings);
}

void Cpm::setChemotaxis(int type, double lambda){
    _settings.types[type].chemotaxis = lambda;
    setSettings(_settings);
}

void Cpm::addCell(int type, int x, int y) {
    _activeCells++;
    int posIndex = calculateIndex(x, y, _dimension);
    _cellIds[posIndex] = _activeCells + (type << 24);
    _types.push_back(type);
    _history.push_back(list<v3>());
}

void Cpm::addCell(int type, int x, int y, int z) {
    _activeCells++;
    int posIndex = calculateIndex(x, y, z, _dimension);
    _cellIds[posIndex] = _activeCells + (type << 24);
    _types.push_back(type);
    _history.push_back(list<v3>());
}

Cpm::Cpm(int dimension, int nrOfCells, int dimensionality, int temperature, bool hasField, int history, bool hasAct): 
    _dimension(dimension), _nrOfCells(nrOfCells), _activeCells(0) , 
    _dimensionality(dimensionality), _mcs(0), _hasField(hasField), 
    _historySize(history), _hasAct(hasAct) {
        _activeHistory = 0;
        _latticePoints = 1;
        for (int i = 0; i < _dimensionality; i++) {
            _latticePoints *= _dimension;
        }
        memset(&_settings, 0, sizeof(Settings));

        _settings.temperature = temperature;

        if (_hasField)
            _field = new double[_latticePoints * _dimensionality]();
        _cellIds = new unsigned int[_latticePoints]();
        if (_hasAct)
            _acts = new unsigned int[_latticePoints]();
        _areas = new unsigned int[nrOfCells+1]();
        _circumferences = new unsigned int[nrOfCells+1]();
        _centroids = new unsigned long long int[_dimensionality*(nrOfCells+1)]();
        _prefDirs = new double[_dimensionality*(nrOfCells+1)]();

        if (_hasField)
            handleCudaError(cudaMalloc(&_d_field, 
                        _dimensionality*_latticePoints*sizeof(double)));
        handleCudaError(cudaMalloc(&_d_cellIds, 
                    _latticePoints*sizeof(unsigned int)));
        if (_hasAct)
            handleCudaError(cudaMalloc(&_d_acts, 
                        _latticePoints*sizeof(unsigned int)));
        handleCudaError(cudaMalloc(&_d_circumferences, 
                    (nrOfCells+1)*sizeof(unsigned int)));
        handleCudaError(cudaMalloc(&_d_centroids, 
                    _dimensionality*(nrOfCells+1)*sizeof(unsigned long long int)));
        handleCudaError(cudaMalloc(&_d_areas, 
                    (nrOfCells+1)*sizeof(unsigned int)));
        handleCudaError(cudaMalloc(&_d_writeAreas, 
                    (nrOfCells+1)*sizeof(unsigned int)));
        handleCudaError(cudaMalloc(&_d_analytics, 3*sizeof(unsigned int)));

        handleCudaError(cudaMalloc(&_d_types, (nrOfCells+1)*sizeof(int)));

        handleCudaError(cudaMalloc(&_d_prefDirs, 
                    _dimensionality*(nrOfCells+1)*sizeof(double)));

        handleCudaError(cudaMalloc(&_d_history, 
                    _historySize*_dimensionality*(nrOfCells+1)*sizeof(unsigned long long int)));

        random_device rd;
        mt19937 mt(rd());
        normal_distribution<double> distribution(0.0,1.0);

        for (int i = 0; i < nrOfCells + 1; i++) {
            double x = distribution(mt);
            double y = distribution(mt);
            double z = distribution(mt);
            double dist = x*x + y*y;
            if (_dimensionality == 3)
                dist += z*z;
            dist = sqrt(dist);
            x /= dist;
            y /= dist;
            z /= dist;


            _prefDirs[i*_dimensionality] = x;
            _prefDirs[i*_dimensionality+1] = y;

            if (_dimensionality == 3)
                _prefDirs[i*_dimensionality+2] = z;


        }
        handleCudaError(cudaMemcpy(_d_prefDirs, _prefDirs, 
                    _dimensionality*(nrOfCells+1)*sizeof(double) , 
                    cudaMemcpyHostToDevice));
        
        //int posIndex = calculateIndex(dimension/2, dimension/2, dimension);
        //_cellIds[posIndex] = 1 + (1 << 24);

        setSettings(_settings);

        _types.push_back(0);
        //cudaFuncSetAttribute(copyAttempt, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
    }

Cpm::~Cpm() {
    if (_hasField)
        delete[] _field;
    delete[] _cellIds;
    if (_hasAct)
        delete[] _acts;
    delete[] _areas;
    delete[] _centroids;
    delete[] _circumferences;

    if (_hasField)
      handleCudaError(cudaFree(_d_field));
    handleCudaError(cudaFree(_d_cellIds));
    if (_hasAct)
        handleCudaError(cudaFree(_d_acts));
    handleCudaError(cudaFree(_d_circumferences));
    handleCudaError(cudaFree(_d_areas));
    handleCudaError(cudaFree(_d_centroids));
    handleCudaError(cudaFree(_d_writeAreas));
    handleCudaError(cudaFree(_d_analytics));
    handleCudaError(cudaFree(_d_types));
    handleCudaError(cudaFree(_d_history));
}

int Cpm::getDimension() {
    return _dimension;
}

void Cpm::run(
        int cellSync, int blockSync, int globalSync, 
        int threadsPerBlock, int positionsPerThread, int positionsPerCheckerboard,
        int updatesPerCheckerboardSwitch, int updatesPerBarrier,
        int iterations, int innerIterations, bool sharedToggle, bool partialDispatch, bool centroidTracking
        ) {




    //cout << "shared: " << sharedToggle << endl;
    //cout << "partial dispatch: " << partialDispatch << endl;
    //bool sharedToggle = true;
    bool actToggle = !partialDispatch;
    bool persistenceToggle = !partialDispatch;
    bool perimeterToggle = !partialDispatch;
    bool chemotaxisToggle = !partialDispatch;
    bool fixedToggle = !partialDispatch;

    for (int i = 0; i < numTypes; i++) {
        if (_settings.types[i].actLambda != 0)
            actToggle = true;
        if (_settings.types[i].perimeterLambda != 0)
            perimeterToggle = true;
        if (_settings.types[i].persistenceLambda != 0)
            persistenceToggle = true;
        if (_settings.types[i].chemotaxis != 0)
            chemotaxisToggle = true;
        if (_settings.types[i].fixed)
            fixedToggle = true;
    }

    void* arguments[26];
    bool insideThreadLoop = 0;
    arguments[0] = &_d_cellIds;
    arguments[1] = &_d_areas;
    arguments[2] = &_d_writeAreas;
    arguments[3] = &_d_circumferences;
    arguments[4] = &_d_centroids;
    arguments[5] = (void*)&_dimension;

    arguments[7] = &_d_acts;
    arguments[9] = &_d_analytics;


    arguments[10] = (void*)&innerIterations;
    arguments[11] = (void*)&positionsPerThread;
    arguments[12] = (void*)&positionsPerCheckerboard;
    arguments[13] = (void*)&updatesPerCheckerboardSwitch;
    arguments[14] = (void*)&updatesPerBarrier;
    arguments[15] = (void*)&blockSync;
    arguments[16] = (void*)&globalSync;
    arguments[17] = (void*)&cellSync;
    arguments[18] = (void*)&insideThreadLoop;


    random_device rd;
    mt19937 mt(rd());
    uniform_int_distribution<int> dist(1, 2147483646);
    int seed = dist(mt);
    arguments[19] = &_d_field;
    arguments[20] = &_d_prefDirs;
    int sharedDim = 64;
    if (_dimensionality == 3)
        sharedDim = 32;
    arguments[21] = &sharedDim;



    arguments[6] = &seed;


    int blocksInGrid = _dimension / threadsPerBlock / positionsPerThread;


    uint32_t device_id = 0;

    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, device_id);
    int sm_count = devProp.multiProcessorCount;

    //cout << "sm count: " << sm_count << endl;
    //cout << "multiprocessor count: " << devProp.multiProcessorCount<< endl;
    //cout << "shared mem/SM: " << devProp.sharedMemPerMultiprocessor << endl;

    //int numBlocksPerSm;
    //int numThreads = 
    //cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, kernel, numThreads, 0);


    int minGridSize;
    int blockSize;
    int numBlocksPerSm;


    cudaError_t err = cudaGetLastError();
    handleCudaError(err);
    int coopAttrib;
    cudaDeviceGetAttribute(&coopAttrib, cudaDevAttrCooperativeLaunch, device_id);

    int globalX = 0;
    int globalY = 0;
    int globalZ = 0;
    bool coopSwitch = coopAttrib==1;
    bool globalCheckerboardSwitch = coopAttrib==0;

    arguments[22] = &globalCheckerboardSwitch;
    arguments[23] = &globalX;
    arguments[24] = &globalY;
    arguments[25] = &globalZ;

    int checkerBoards = positionsPerThread / positionsPerCheckerboard;


    for (int mcs = 0; mcs < iterations/innerIterations; mcs++) {
        arguments[8] = &_mcs;
        if (_dimensionality == 2) {
            int checkerboardMultiplier = (positionsPerCheckerboard * positionsPerCheckerboard)/updatesPerCheckerboardSwitch;
            dim3 gridDims(blocksInGrid, blocksInGrid);
            dim3 blockDims(threadsPerBlock, threadsPerBlock);

            auto kernelPointer = get_2d_kernel_pointer(sharedToggle, actToggle, perimeterToggle, persistenceToggle, chemotaxisToggle, fixedToggle);
            if (sharedToggle) {
                cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernelPointer, 64*64*sizeof(unsigned int), 8*8);
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, kernelPointer, blockSize, 64*64*sizeof(unsigned int));
                //cout << "min grid size: " << minGridSize << " block size: " << blockSize << " num blocks/sm: " << numBlocksPerSm << endl;
                cudaLaunchCooperativeKernel((void*)kernelPointer, gridDims, blockDims, arguments, 64*64*sizeof(unsigned int));


            } else {
                if (coopAttrib) {
                    cudaLaunchCooperativeKernel((void*)kernelPointer, gridDims, blockDims, arguments);
                } else {
        for (int outer = 0; outer < checkerboardMultiplier; outer++) {
            for(int globalX = 0; globalX < checkerBoards; globalX++) {
            for(int globalY = 0; globalY < checkerBoards; globalY++) {
                    kernelPointer<<<gridDims, blockDims>>>(
                            _d_cellIds, _d_areas, _d_writeAreas, 
                            _d_circumferences, _d_centroids, _dimension, seed, 
                            _d_acts, _mcs, _d_analytics, innerIterations,
                            positionsPerThread, positionsPerCheckerboard,
                            updatesPerCheckerboardSwitch, updatesPerBarrier, 
                            blockSync, globalSync, cellSync, insideThreadLoop, 
                            _d_field, _d_prefDirs, sharedDim, coopSwitch,
                            globalX, globalY);
            }}}
                }
            }
        } else if (_dimensionality == 3) {
            int checkerboardMultiplier = (positionsPerCheckerboard * positionsPerCheckerboard * positionsPerCheckerboard)/updatesPerCheckerboardSwitch;
            dim3 gridDims(blocksInGrid, blocksInGrid, blocksInGrid);
            dim3 blockDims(threadsPerBlock, threadsPerBlock, threadsPerBlock);

            //cout << "grid dim: " << blocksInGrid << "(" << blocksInGrid * blocksInGrid * blocksInGrid << ")" << endl;
            //cout << "block dim: " << threadsPerBlock << "(" << threadsPerBlock * threadsPerBlock * threadsPerBlock << ")" << endl;

            auto kernelPointer = get_3d_kernel_pointer(sharedToggle, actToggle, perimeterToggle, persistenceToggle, chemotaxisToggle, fixedToggle, centroidTracking);

//cudaFuncSetAttribute(kernelPointer, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
            //cudaFuncSetAttribute(kernelPointer, cudaFuncAttributeMaxDynamicSharedMemorySize, 128*1024);
            //cudaFuncAttributes attribs;
            //cudaFuncGetAttributes(&attribs, kernelPointer);
            //cout << "preferred carveout size: " << attribs.preferredShmemCarveout << endl;
            //cout << "shared size: " << attribs.sharedSizeBytes << endl;



	    
            if (sharedToggle) {
                cudaLaunchCooperativeKernel((void*)kernelPointer, gridDims, blockDims, arguments, 32*32*32*sizeof(unsigned int));
            } else {
                if (coopAttrib) {
                    cudaLaunchCooperativeKernel((void*)kernelPointer, gridDims, blockDims, arguments);
                } else {
        for (int outer = 0; outer < checkerboardMultiplier; outer++) {
            for(int globalX = 0; globalX < checkerBoards; globalX++) {
            for(int globalY = 0; globalY < checkerBoards; globalY++) {
            for(int globalZ = 0; globalZ < checkerBoards; globalZ++) {
                    kernelPointer<<<gridDims, blockDims>>>(
                            _d_cellIds, _d_areas, _d_writeAreas, 
                            _d_circumferences, _d_centroids, _dimension, seed, 
                            _d_acts, _mcs, _d_analytics, innerIterations,
                            positionsPerThread, positionsPerCheckerboard,
                            updatesPerCheckerboardSwitch, updatesPerBarrier, 
                            blockSync, globalSync, cellSync, insideThreadLoop, 
                            _d_field, _d_prefDirs, sharedDim, coopSwitch,
                            globalX, globalY, globalZ);
            }}}}
                }
            }
        }
        cudaError_t err = cudaGetLastError();
        handleCudaError(err);
        _mcs += innerIterations;


        if (_dimensionality == 3) {

            //updatePrefDir();

            cudaDeviceSynchronize();
            // register centroid in history list
            cudaMemcpy(&_d_history[_dimensionality*(_nrOfCells+1)*_activeHistory], 
                    _d_centroids, 
                    _dimensionality*(_nrOfCells+1)*sizeof(unsigned long long int), 
                    cudaMemcpyDeviceToDevice);

            // move index in history circle buffer
            _activeHistory = (_activeHistory+1) % _historySize;



            int gridSize = ((_activeCells+1)-1)/1024 + 1;
            int threadsPerBlock = (_activeCells+1) > 1024 ? 1024 : _activeCells+1;

            updatePrefDirKernel<<<gridSize, threadsPerBlock>>>(
                    _d_prefDirs, _d_types, _d_centroids, 
                    &_d_history[_dimensionality * (_nrOfCells+1)*_activeHistory], 
                    _activeCells, _dimension);




        }
    }
    //_mcs += iterations / innerIterations;
}

unsigned int* Cpm::getCellIds() {
    return _cellIds;
}

unsigned int* Cpm::getActData() {
    return _acts;
}

double* Cpm::getField() {
    return _field;
}

void Cpm::retreiveCellIdsFromGpu() {
    cudaMemcpy(_cellIds, _d_cellIds, _latticePoints*sizeof(unsigned int), 
            cudaMemcpyDeviceToHost);
    if (_hasAct)
        cudaMemcpy(_acts, _d_acts, _latticePoints*sizeof(unsigned int), 
            cudaMemcpyDeviceToHost);
    if (_hasField)
        cudaMemcpy(_field, _d_field, _dimensionality*_latticePoints*sizeof(double), 
                cudaMemcpyDeviceToHost);
}
void Cpm::moveCellIdsToGpu() {
    if (_hasField)
    handleCudaError(cudaMemcpy(_d_field, _field, 
                _dimensionality*_latticePoints*sizeof(double), cudaMemcpyHostToDevice));
    handleCudaError(cudaMemcpy(_d_cellIds, _cellIds, 
                _latticePoints*sizeof(unsigned int), cudaMemcpyHostToDevice));
    if (_hasAct)
        handleCudaError(cudaMemcpy(_d_acts, _acts, 
                    _latticePoints*sizeof(unsigned int), cudaMemcpyHostToDevice));

    handleCudaError(cudaMemcpy(_d_types, &_types[0], 
                _types.size()*sizeof(int), cudaMemcpyHostToDevice));

    handleCudaError(cudaMemset(_d_circumferences, 0, 
                (_nrOfCells+1)*sizeof(unsigned int)));
    handleCudaError(cudaMemset(_d_areas, 0, (_nrOfCells+1)*sizeof(unsigned int)));
    handleCudaError(cudaMemset(_d_writeAreas, 0, (_nrOfCells+1)*sizeof(unsigned int)));
    handleCudaError(cudaMemset(_d_centroids, 0, _dimensionality*(_nrOfCells+1)*sizeof(unsigned long long int)));
    handleCudaError(cudaMemset(_d_analytics, 0, 3*sizeof(unsigned int)));

    if (_dimensionality == 2) {
    dim3 gridDims(_dimension, max(_dimension / 1024, 1));
    int threadsPerBlock = _dimension > 1024 ? 1024 : _dimension;

    calculateAreasAndCircumferences<<<gridDims, threadsPerBlock>>>(_d_cellIds, 
            _d_areas, _d_writeAreas, _d_circumferences, _d_centroids, _dimension);
    cudaError_t err = cudaGetLastError();
    handleCudaError(err);
    } else if (_dimensionality == 3) {
    dim3 gridDims(_dimension, _dimension, max(_dimension / 1024, 1));
    int threadsPerBlock = _dimension > 1024 ? 1024 : _dimension;

    calculateAreasAndCircumferences3d<<<gridDims, threadsPerBlock>>>(_d_cellIds, 
            _d_areas, _d_writeAreas, _d_circumferences, _d_centroids, _dimension);

    }

    for (int i = 0; i < _historySize; i++) {
        cudaMemcpy(&_d_history[_dimensionality*(_nrOfCells+1)*i], 
                _d_centroids, 
                _dimensionality*(_nrOfCells+1)*sizeof(unsigned long long int), 
                cudaMemcpyDeviceToDevice);
    }


    cudaError_t err = cudaGetLastError();
    handleCudaError(err);

    //Volta supports shared memory capacities of 0, 8, 16, 32, 64, or 96 KB per SM.
    //cudaFuncSetAttribute(copyAttempt<false>, cudaFuncAttributePreferredSharedMemoryCarveout, 0);
    //cudaFuncAttributes attribs;
    //cudaFuncGetAttributes(&attribs, copyAttempt<false>);
    //cout << "preferred carveout size: " << attribs.preferredShmemCarveout << endl;
    //cout << "shared size: " << attribs.sharedSizeBytes << endl;


}

void Cpm::updatePrefDir() {
    cudaDeviceSynchronize();
    auto currentCentroids = get3dCentroids();


    for (int i = 0; i < currentCentroids.size(); i++) {
        const auto type = _types[i+1];
        if (_settings.types[type].persistenceTime > 0) {
            while (_history[i].size() >= _settings.types[type].persistenceTime) {
                _history[i].pop_front();
            }
        } else {
            while (_history[i].size() > 0) {
                _history[i].pop_front();
            }
        }
        _history[i].push_back(currentCentroids[i]);
    }



    for (int i = 1; i < _activeCells+1; i++) {
        v3 currentPrefDir;
        currentPrefDir.x = _prefDirs[i*3 + 0];
        currentPrefDir.y = _prefDirs[i*3 + 1];
        currentPrefDir.z = _prefDirs[i*3 + 2];

        auto currentDir = currentCentroids[i-1].sub(_history[i-1].front());
        if (currentDir.length() == 0) {
            continue;
        }

        currentDir = currentDir.wrap(_dimension);
        currentDir = currentDir.normalize();

        const auto type = _types[i];
        const auto persistence = _settings.types[type].persistenceDiffusion;

       
        currentPrefDir = currentDir.times(1-persistence).add(currentPrefDir.times(persistence)).normalize();
        _prefDirs[i*3 + 0] = currentPrefDir.x;
        _prefDirs[i*3 + 1] = currentPrefDir.y;
        _prefDirs[i*3 + 2] = currentPrefDir.z;

    }

    handleCudaError(cudaMemcpy(_d_prefDirs, _prefDirs, 
                    _dimensionality*(_nrOfCells+1)*sizeof(double) , 
                    cudaMemcpyHostToDevice));
}




void Cpm::synchronize() {
    cudaDeviceSynchronize();
}
