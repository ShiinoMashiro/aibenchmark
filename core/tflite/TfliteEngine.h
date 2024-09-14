#ifndef TFLITE_H
#define TFLITE_H

#include "Engine.h"
#include "tflite/c_api.h"
#include "tflite/delegate.h"
// #include "xnnpack_delegate.h"

class TfliteEngine : public ENGINE::Engine {
    public:

    virtual int init(const void* modelBuffer, int modelSize, ENGINE::Options options) override;

    virtual int resizeInput(int inputIndex, const std::vector<int> &inputShape) override;

    virtual int resizeEngine() override;

    virtual int setInput(const float* inputBuffer) override;

    virtual int inference() override;

    virtual int getOutput(float* outputBuffer) override;

    virtual std::vector<int> getInputShape() override;

    virtual std::vector<int> getOutputShape() override;

    virtual const char* getVersion() override;

    private:
    TfLiteInterpreter* interpreter_ = nullptr;
    TfLiteModel* model_ = nullptr;
    TfLiteInterpreterOptions* options_ = nullptr;
    TfLiteDelegate* delegate_ = nullptr;
};

#endif