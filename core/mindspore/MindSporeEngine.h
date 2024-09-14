#ifndef MINDSPORE_H
#define MINDSPORE_H

#include "Engine.h"
#include "mindspore/context.h"
#include "mindspore/model.h"
#include "mindspore/status.h"
#include "mindspore/types.h"

class MindSporeEngine : public ENGINE::Engine {
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
    std::shared_ptr<mindspore::Model> model_;
};

#endif