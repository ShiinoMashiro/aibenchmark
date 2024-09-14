#include <iostream>
#include "MindSporeEngine.h"

int MindSporeEngine::init(const void* modelBuffer, int modelSize, ENGINE::Options options) {
    // Create and init context, add CPU device info
    auto context = std::make_shared<mindspore::Context>();
    if (context == nullptr) {
        std::cerr << "New context failed." << std::endl;
        return -1;
    }
    context->SetThreadNum(options.threads);

    auto &deviceList = context->MutableDeviceInfo();
    if (options.backend == ENGINE::Backend::gpu) {
        auto gpuDeviceInfo = std::make_shared<mindspore::GPUDeviceInfo>();
        gpuDeviceInfo->SetEnableFP16(false);
        deviceList.push_back(gpuDeviceInfo);
    } else {
        auto cpuDeviceInfo = std::make_shared<mindspore::CPUDeviceInfo>();
        cpuDeviceInfo->SetEnableFP16(false);
        deviceList.push_back(cpuDeviceInfo);
    }
    
    // Create model
    model_ = std::shared_ptr<mindspore::Model>(new(std::nothrow) mindspore::Model());
    if (model_ == nullptr) {
        std::cerr << "New Model failed." << std::endl;
        return -1;
    }

    // Build model
    auto buildRet = model_->Build(modelBuffer, modelSize, mindspore::kMindIR, context);
    if (buildRet != mindspore::kSuccess) {
        std::cerr << "Build model error " << buildRet << std::endl;
        return -1;
    }

    return 0;
}

int MindSporeEngine::resizeInput(int inputIndex, const std::vector<int> &inputShape) {
    // Resizes the specified input tensor.
    auto input = model_->GetInputs()[0];
    model_->Resize({input}, {std::vector<int64_t>(inputShape.begin(), inputShape.end())});
    return 0;
}

int MindSporeEngine::resizeEngine() {
    return 0;
}

int MindSporeEngine::setInput(const float* inputBuffer) {
    auto input = model_->GetInputs()[0];
    auto inputSizeByBytes = input.DataSize();
    memcpy(input.MutableData(), inputBuffer, inputSizeByBytes);
    return 0;
}

int MindSporeEngine::inference() {
    // Run the interpreter
    auto inputs = model_->GetInputs();
    auto outputs = model_->GetOutputs();
    auto status = model_->Predict(inputs, &outputs);
    if (status != mindspore::kSuccess) {
        printf("Something went wrong when running the mindspore model.\n");
        return -5;
    }
    return 0;
}

int MindSporeEngine::getOutput(float* outputBuffer) {
    auto output = model_->GetOutputs()[0];
    auto outputSizeByBytes = output.DataSize();
    memcpy(outputBuffer, output.MutableData(), outputSizeByBytes);
    return 0;
}

std::vector<int> MindSporeEngine::getInputShape() {
    auto input = model_->GetInputs()[0];
    auto shape = input.Shape();
    return std::vector<int>(shape.begin(), shape.end());
}

std::vector<int> MindSporeEngine::getOutputShape() {
    auto output = model_->GetOutputs()[0];
    auto shape = output.Shape();
    return std::vector<int>(shape.begin(), shape.end());
}

const char* MindSporeEngine::getVersion() {
    return mindspore::Version().c_str();
}

