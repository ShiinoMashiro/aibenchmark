#ifndef ENGINE_FACTORY_H
#define ENGINE_FACTORY_H

#include "Engine.h"
#include "TfliteEngine.h"
#include "MnnEngine.h"
#include "MindSporeEngine.h"

class EngineFactory {
    public:
    static std::shared_ptr<ENGINE::Engine> create(std::string name) {
        if (name == "tflite") {
            return std::shared_ptr<ENGINE::Engine>(new TfliteEngine());
        } else if (name == "mnn") {
            return std::shared_ptr<ENGINE::Engine>(new MnnEngine());
        } else if (name == "mindspore") {
            return std::shared_ptr<ENGINE::Engine>(new MindSporeEngine());
        } else {
            printf("Such engine: %s not support.", name.c_str());
            return nullptr;
        }
    }
};

#endif