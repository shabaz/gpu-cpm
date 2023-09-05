#include <Python.h>
#include <numpy/arrayobject.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <iostream>

using namespace std;

#include "cpm.h"

typedef struct {
    PyObject_HEAD
    Cpm* ptrObj;
} PyCpm;



static PyModuleDef gpucpmmodule = {
    PyModuleDef_HEAD_INIT,
    "gpucpm",
    "GPU CPM Python wrapper",
    -1,
    NULL, NULL, NULL, NULL, NULL
};

static int PyCpm_init(PyCpm *self, PyObject* args, PyObject* kwds) {
    int dimension;
    int numberOfCells;
    int dimensionality;
    int temperature;
    int hasField;
    int hasAct = 1;
    int history = 1;
    if (! PyArg_ParseTuple(args, "iiiip|ip", &dimension, &dimensionality, 
                &numberOfCells, &temperature, &hasField, &history, &hasAct))
        return -1;

    self->ptrObj = new Cpm(dimension, numberOfCells, dimensionality, temperature, hasField, history, hasAct);
    return 0;
}

static void PyCpm_dealloc(PyCpm* self) {
    delete self->ptrObj;
    Py_TYPE(self)->tp_free(self);
}



static PyTypeObject PyCpmType = { PyVarObject_HEAD_INIT(NULL, 0)
                                    "gpucpm.Cpm"   /* tp_name */
                                };


static PyObject * PyCpm_setConstraints(PyCpm* self, PyObject* args, 
        PyObject* kwargs )
{
    char* keywords [] = {
        "cell_type", 
        "other_cell_type", 
        "lambda_perimeter", 
        "target_perimeter",
        "lambda_area", 
        "target_area",
        "lambda_act",
        "max_act",
        "adhesion",
        "fixed",
        "lambda_chemotaxis",
        "lambda_persistence", 
        "persistence_diffusion", 
        "persistence_time",
        "set_adhesion_symmetric",
        "stickiness",
        NULL
    };
    int cellType = -1, targetPerimeter = -1, targetArea = -1, maxAct = -1, 
        otherCellType = -1, adhesion = -1, fixed = -1, persistenceTime = -1, stickiness = -1;
    double lambdaPerimeter = -1, lambdaArea = -1, lambdaAct = -1, 
           lambdaChemotaxis = -1, lambdaPersistence = -1, 
           persistenceDiffusion = -1, setAdhesionSymmetric = 1;
     if (! PyArg_ParseTupleAndKeywords(args, kwargs, "i|$idididiiidddiii", 
                 keywords, &cellType, &otherCellType, &lambdaPerimeter, 
                 &targetPerimeter, &lambdaArea, &targetArea, &lambdaAct, 
                 &maxAct, &adhesion, &fixed, &lambdaChemotaxis, 
                 &lambdaPersistence, &persistenceDiffusion, &persistenceTime, &setAdhesionSymmetric, &stickiness))
         return Py_False;

     if (fixed >= 0) {
         (self->ptrObj)->setFixed(cellType, fixed);
     }

     if (lambdaPersistence != -1 && persistenceDiffusion != -1 && 
             persistenceTime != -1) {
        (self->ptrObj)->setPersistence(cellType, persistenceTime, 
                persistenceDiffusion, lambdaPersistence);
     }

     if (lambdaChemotaxis != -1) {
         (self->ptrObj)->setChemotaxis(cellType, lambdaChemotaxis);
     }


     if (lambdaPerimeter != -1 && targetPerimeter != -1) {
         (self->ptrObj)->setPerimeter(cellType, targetPerimeter, 
                 lambdaPerimeter);
     }

     if (lambdaArea != -1 && targetArea != -1) {
         (self->ptrObj)->setArea(cellType, targetArea, lambdaArea);
     }

     if (lambdaAct != -1 && maxAct != -1) {
         (self->ptrObj)->setAct(cellType, maxAct, lambdaAct);
     }

     if (otherCellType != -1 && adhesion != -1) {
         (self->ptrObj)->setAdhesion(cellType, otherCellType, 
                 adhesion, setAdhesionSymmetric);
     }

     if (otherCellType != -1 && stickiness != -1) {
         (self->ptrObj)->setStickiness(cellType, otherCellType, 
                 stickiness);
     }

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject * PyCpm_addCell(PyCpm* self, PyObject* args)
{
    int x;
    int y;
    int z = -1;
    int type;
    if (! PyArg_ParseTuple(args, "iii|i", &type, &x, &y, &z))
        return Py_False;

    if (z == -1) {
        (self->ptrObj)->addCell(type, x, y);
    } else {
        (self->ptrObj)->addCell(type, x, y, z);
    }

    Py_INCREF(Py_None);
    return Py_None;
}


static PyObject * PyCpm_getCentroids(PyCpm* self, PyObject* args)
{
    auto dimensionality = (self->ptrObj)->getDimensionality();
   
    if (dimensionality == 2) {
        auto centroids = (self->ptrObj)->get2dCentroids();
        npy_intp const dims[2] = {int(centroids.size()), 2};
        PyArrayObject* output = (PyArrayObject*) PyArray_SimpleNew(2, dims, PyArray_DOUBLE);
        double* data = (double*) output->data;
        for(int i = 0; i < centroids.size(); i++) {
            data[i*2 + 0] = centroids[i].x;
            data[i*2 + 1] = centroids[i].y;
        }
        return (PyObject*)output;
    } else {
        auto centroids = (self->ptrObj)->get3dCentroids();
        npy_intp const dims[2] = {int(centroids.size()), 3};
        PyArrayObject* output = (PyArrayObject*) PyArray_SimpleNew(2, dims, PyArray_DOUBLE);
        double* data = (double*) output->data;
        for(int i = 0; i < centroids.size(); i++) {
            data[i*3 + 0] = centroids[i].x;
            data[i*3 + 1] = centroids[i].y;
            data[i*3 + 2] = centroids[i].z;
        }
        return (PyObject*)output;
    }
}




static PyObject * PyCpm_run(PyCpm* self, PyObject* args, PyObject* kwargs )
{


    char* keywords [] = {
        "cell_sync",
        "block_sync",
        "global_sync",

        "threads_per_block",
        "positions_per_thread",
        "positions_per_checkerboard", 

        "updates_per_checkerboard_switch",
        "updates_per_barrier",

        "iterations", 
        "inner_iterations", 

        "shared",
        "partial_dispatch",

        "centroid_tracking",
        NULL
    };
        int cellSync = -1, blockSync = -1, globalSync = -1, 
        threadsPerBlock = -1, positionsPerThread = -1, positionsPerCheckerboard = -1, 
        updatesPerBarrier = -1, updatesPerCheckerboardSwitch = -1, 
        iterations = -1, innerIterations = -1, sharedToggle = 0, partialDispatch = 0, centroidTracking = 0;
     if (! PyArg_ParseTupleAndKeywords(args, kwargs, "iiiiiiiiii|iip", 
                 keywords, 
                 &cellSync, &blockSync, &globalSync, 
                 &threadsPerBlock, &positionsPerThread, &positionsPerCheckerboard,
                 &updatesPerCheckerboardSwitch, &updatesPerBarrier, 
                 &iterations, &innerIterations, &sharedToggle, &partialDispatch, &centroidTracking))
         return Py_False;


    (self->ptrObj)->run(cellSync, blockSync, globalSync,
            threadsPerBlock, positionsPerThread, positionsPerCheckerboard, 
            updatesPerCheckerboardSwitch, updatesPerBarrier, 
            iterations, innerIterations, sharedToggle, partialDispatch, centroidTracking);

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject * PyCpm_runAsync(PyCpm* self, PyObject* args)
{
    int ticks;

    if (! PyArg_ParseTuple(args, "i", &ticks))
        return Py_False;

    //(self->ptrObj)->runAsync(ticks);

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject * PyCpm_join(PyCpm* self, PyObject* args)
{
    //(self->ptrObj)->join();

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject * PyCpm_pushToGpu(PyCpm* self, PyObject* args)
{
    (self->ptrObj)->moveCellIdsToGpu();

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject * PyCpm_synchronize(PyCpm* self, PyObject* args)
{
    (self->ptrObj)->synchronize();

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject * PyCpm_pullFromGpu(PyCpm* self, PyObject* args)
{
    (self->ptrObj)->retreiveCellIdsFromGpu();

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject * PyCpm_getField(PyCpm* self, PyObject* args)
{
    int dimension = (self->ptrObj)->getDimension();
    auto dimensionality = (self->ptrObj)->getDimensionality();

    if (dimensionality == 2) {
        npy_intp shape[] = {2, dimension, dimension};
        PyObject* arr = PyArray_SimpleNewFromData(3, shape, NPY_DOUBLE, 
                (self->ptrObj)->getField());
        return arr;
    } else {
        npy_intp shape[] = {3, dimension, dimension, dimension};
        PyObject* arr = PyArray_SimpleNewFromData(4, shape, NPY_DOUBLE, 
                (self->ptrObj)->getField());
        return arr;
    }
}

static PyObject * PyCpm_getState(PyCpm* self, PyObject* args)
{
    int dimension = (self->ptrObj)->getDimension();
    auto dimensionality = (self->ptrObj)->getDimensionality();

    if (dimensionality == 2) {
        npy_intp shape[] = {dimension, dimension};
        PyObject* arr = PyArray_SimpleNewFromData(2, shape, NPY_INT, 
                (self->ptrObj)->getCellIds());
        return arr;
    } else {
        npy_intp shape[] = {dimension, dimension, dimension};
        PyObject* arr = PyArray_SimpleNewFromData(3, shape, NPY_INT, 
                (self->ptrObj)->getCellIds());
        return arr;
    }
}



static PyObject * PyCpm_getActState(PyCpm* self, PyObject* args)
{
    int dimension = (self->ptrObj)->getDimension();
    auto dimensionality = (self->ptrObj)->getDimensionality();

    if (dimensionality == 2) {
        npy_intp shape[] = {dimension, dimension};
        PyObject* arr = PyArray_SimpleNewFromData(2, shape, NPY_INT, 
                (self->ptrObj)->getActData());
        return arr;
    } else {
        npy_intp shape[] = {dimension, dimension, dimension};
        PyObject* arr = PyArray_SimpleNewFromData(3, shape, NPY_INT, 
                (self->ptrObj)->getActData());
        return arr;
    }
}

/*
static PyObject * PyCpm_initializeFromArray(PyCpm* self, PyObject* args)
{
    PyObject *arg=NULL;
    int count;
    if (!PyArg_ParseTuple(args, "Oi", &arg, &count)) return NULL;
    int dims = PyArray_NDIM(arg);
    npy_intp* dim_vals = PyArray_DIMS(arg);
    int x = dim_vals[0];
    int y = dim_vals[1];

    int dimension = (self->ptrObj)->getDimension();

    int xoffset = (dimension - x)/2;
    int yoffset = (dimension - y)/2;

    long* data = static_cast<long *>(PyArray_DATA(arg));

    for (int i = 0; i < x; i++) {
        for (int j = 0; j < y; j++) {
            long cellId = *static_cast<long*>(PyArray_GETPTR2(arg, i, j));
            if (cellId != 0) {
                (self->ptrObj)->setPoint(j+yoffset, i+xoffset, cellId, 1);
            }
        }
    }
    (self->ptrObj)->updateCellProps(count);

    Py_INCREF(Py_None);
    return Py_None;
}
*/



static PyMethodDef PyCpm_methods[] = {
    //{ "initialize_from_array", (PyCFunction)PyCpm_initializeFromArray, METH_VARARGS, "initialize simulation from array" },
    { "run", (PyCFunction)PyCpm_run, METH_VARARGS | METH_KEYWORDS, "run for certain number of ticks" },
    //{ "run_async", (PyCFunction)PyCpm_runAsync, METH_VARARGS, "run for certain number of ticks in seperate thread" },
    { "push_to_gpu", (PyCFunction)PyCpm_pushToGpu, METH_VARARGS, "move data from cpu memory to gpu" },
    { "pull_from_gpu", (PyCFunction)PyCpm_pullFromGpu, METH_VARARGS, "move data from gpu memory to cpu" },
    { "get_state", (PyCFunction)PyCpm_getState, METH_VARARGS, "get state of CPM lattice" },
    { "get_field", (PyCFunction)PyCpm_getField, METH_VARARGS, "get chemotaxis field of CPM" },
    { "add_cell", (PyCFunction)PyCpm_addCell, METH_VARARGS, "add cell of type at location" },
    { "get_centroids", (PyCFunction)PyCpm_getCentroids, METH_VARARGS, "get centroids of cells" },
    { "synchronize", (PyCFunction)PyCpm_synchronize, METH_VARARGS, "make sure kernel has finished" },

    { "get_act_state", (PyCFunction)PyCpm_getActState, METH_VARARGS, "get state of CPM act lattice" },
    { "set_constraints", (PyCFunction)PyCpm_setConstraints, METH_VARARGS | METH_KEYWORDS, "set CPM hamiltonian constraints" },
    {NULL}  /* Sentinel */
};



PyMODINIT_FUNC PyInit_gpucpm(void)
{
    import_array();

    PyObject* m;

    PyCpmType.tp_new = PyType_GenericNew;
    PyCpmType.tp_basicsize=sizeof(PyCpm);
    PyCpmType.tp_dealloc=(destructor) PyCpm_dealloc;
    PyCpmType.tp_flags=Py_TPFLAGS_DEFAULT;
    PyCpmType.tp_doc="CPM objects";
    PyCpmType.tp_methods=PyCpm_methods;
    //~ PyVoiceType.tp_members=Noddy_members;
    PyCpmType.tp_init=(initproc)PyCpm_init;

    if (PyType_Ready(&PyCpmType) < 0)
        return NULL;

    m = PyModule_Create(&gpucpmmodule);
    if (m == NULL)
        return NULL;

    Py_INCREF(&PyCpmType);
    PyModule_AddObject(m, "Cpm", (PyObject *)&PyCpmType); // Add Voice object to the module
    return m;
}
