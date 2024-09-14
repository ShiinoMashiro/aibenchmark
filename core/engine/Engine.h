#ifndef ENGINE_H
#define ENGINE_H

#include <vector>

namespace ENGINE {

enum Backend {
    cpu,
    gpu,
    dsp,
    npu
};

struct Options {
    Backend backend;
    int threads;
};

class Engine {
    public:
    
    virtual int init(const void* modelBuffer, int modelSize, ENGINE::Options options) = 0;
    
    virtual int resizeInput(int inputIndex, const std::vector<int> &inputShape) = 0;

    virtual int resizeEngine() = 0;

    virtual int setInput(const float* inputBuffer) = 0;

    virtual int inference() = 0;

    virtual int getOutput(float* outputBuffer) = 0;

    virtual std::vector<int> getInputShape() = 0;

    virtual std::vector<int> getOutputShape() = 0;

    virtual const char* getVersion() = 0;
};

} // namespace ENGINE

#endif