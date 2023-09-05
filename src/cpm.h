#ifndef CPM_H_
#define CPM_H_

#include <vector>
#include <list>
#include <math.h>
#include "settings.h"

struct v2 {
    double x;
    double y;
}
;
struct v3 {
    double x;
    double y;
    double z;

    v3 sub(v3& other) {
        v3 a;
        a.x = x - other.x;
        a.y = y - other.y;
        a.z = z - other.z;
        return a;
    }

    v3 normalize() {
        double l = length();
        v3 a;
        a.x = x / l;
        a.y = y / l;
        a.z = z / l;
        return a;
    }

    double length() {
        return sqrt(x*x + y*y + z*z);
    }

    v3 times(double l) {
        v3 a;
        a.x = x * l;
        a.y = y * l;
        a.z = z * l;
        return a;
    }

    v3 add(v3 b) {
        v3 a;
        a.x = x + b.x;
        a.y = y + b.y;
        a.z = z + b.z;
        return a;
    }


    v3 wrap(int dimension) {
        v3 p = *this;
        if (p.x > dimension/2)
            p.x -= dimension;
        if (p.x < -dimension/2)
            p.x += dimension;
        if (p.y > dimension/2)
            p.y -= dimension;
        if (p.y < -dimension/2)
            p.y += dimension;
        if (p.z > dimension/2)
            p.z -= dimension;
        if (p.z < -dimension/2)
            p.z += dimension;
        return p;
    }



};

class Cpm {
    public:
        Cpm(int dimension, int nrOfCells, int dimensionality, int temperature, bool hasField, int history, bool hasAct);
        ~Cpm();

        void run( int cellSync, int blockSync, int globalSync, 
                int threadsPerBlock, int positionsPerThread, 
                int positionsPerCheckerboard, int updatesPerCheckerboardSwitch, 
                int updatesPerBarrier, int iterations, int innerIterations, 
                bool sharedToggle, bool partialDispatch,
                bool centroidTracking);
        int getDimension();
        unsigned int* getCellIds();
        unsigned int* getActData();
        double* getField();
        std::vector<v2> get2dCentroids();
        std::vector<v3> get3dCentroids();
        void retreiveCellIdsFromGpu();
        void moveCellIdsToGpu();

        void updatePrefDir();

        void setTemperature(int temperature);
        void setPersistence(int type, int time, double diffusion, double lambda);
        void setPerimeter(int type, int target, double lambda);
        void setArea(int type, int target, double lambda);
        void setAct(int type, int maxAct, double lambda);
        void setAdhesion(int type, int otherType, int adhesion, int setBoth);
        void setStickiness(int type, int otherType, int stickiness);
        void setFixed(int type, int fixed);
        void setChemotaxis(int type, double lambda);
        void addCell(int type, int x, int y);
        void addCell(int type, int x, int y, int z);
        int getDimensionality();
        void synchronize();
    private:
        int _activeHistory;
        int _historySize;
        int _mcs;
        int _latticePoints;
        int _dimension;
        int _nrOfCells;
        int _activeCells;
        int _dimensionality;
        bool _hasField;
        bool _hasAct;
        double* _field;
        unsigned int* _cellIds;
        unsigned int* _acts;
        unsigned int* _areas;
        unsigned int* _circumferences;
        unsigned long long int* _centroids;


        std::vector<int> _types;
        //std::vector<v3> _prevCentroids;
        std::vector<std::list<v3>> _history;
        double* _prefDirs;
        unsigned int _analytics[3];

        double* _d_field;
        int* _d_types;
        unsigned int* _d_cellIds;
        unsigned int* _d_acts;
        unsigned int* _d_areas;
        unsigned int* _d_writeAreas;
        unsigned int* _d_analytics;
        unsigned int* _d_circumferences;
        unsigned long long int* _d_centroids;
        unsigned long long int* _d_history;
        double* _d_prefDirs;

        Settings _settings;
};

#endif // CPM_H_

