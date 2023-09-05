#ifndef SETTINGS_H_
#define SETTINGS_H_

const int numTypes = 20;

struct TypeSettings {
    double  persistenceLambda;
    double  persistenceDiffusion;
    int  persistenceTime;

    double  areaLambda;
    int areaTarget;

    double  actLambda;
    int actMax;

    double  perimeterLambda;
    int perimeterTarget;

    short adhesion[numTypes];

    short stickiness[numTypes];

    int fixed;
    double chemotaxis;
};

struct Settings {
    TypeSettings types[numTypes];
    int temperature;
};

#endif // SETTINGS_H_
