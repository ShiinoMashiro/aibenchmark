#include "MnnEngine.h"

using namespace MNN;

int MnnEngine::init(const void* modelBuffer, int modelSize, ENGINE::Options options) {
    // Create interpreter
    interpreter_ = std::shared_ptr<Interpreter> (
        Interpreter::createFromBuffer(modelBuffer, modelSize));
    if (!interpreter_) {
        printf("Failed to create mnn interpreter.\n");
        return -1;
    }

    // Pass through options to mnn config
    ScheduleConfig config;
    if (options.backend == ENGINE::Backend::gpu) {
        config.type = MNN_FORWARD_OPENCL;
    } else {
        config.type = MNN_FORWARD_CPU;
    }
    BackendConfig backendConfig;
    backendConfig.precision = BackendConfig::Precision_Normal;
    config.backendConfig = &backendConfig;
    config.numThread = options.threads;

    // Create session
    session_ = interpreter_->createSession(config);
    if (!session_) {
        printf("Failed to create mnn session.\n");
        return -1;
    }

    return 0;
}

int MnnEngine::resizeInput(int inputIndex, const std::vector<int> &inputShape) {
    // Resizes the specified input tensor.
    auto input = interpreter_->getSessionInput(session_, nullptr);
    interpreter_->resizeTensor(input, inputShape);
    return 0;
}

int MnnEngine::resizeEngine() {
    interpreter_->resizeSession(session_);
    return 0;
}

int MnnEngine::setInput(const float* inputBuffer) {
    auto input = interpreter_->getSessionInput(session_, nullptr);
    auto inputSizeByBytes = input->size();

    // Create host tensor
    std::shared_ptr<Tensor> inputHost(new Tensor(input, Tensor::CAFFE));
    memcpy(inputHost->host<float>(), inputBuffer, inputSizeByBytes);

    // Feed input into model
    input->copyFromHostTensor(inputHost.get());

    return 0;
}

int MnnEngine::inference() {
    // Run the interpreter
    auto errcode = interpreter_->runSession(session_);
    if (errcode != ErrorCode::NO_ERROR) {
        printf("Something went wrong when running the mnn model.\n");
        return -5;
    }
    return 0;
}

int MnnEngine::getOutput(float* outputBuffer) {
    auto output = interpreter_->getSessionOutput(session_, NULL);
    auto outputSizeByBytes = output->size();

    // Create host tensor
    std::shared_ptr<Tensor> outputHost(new Tensor(output, Tensor::CAFFE));
    output->copyToHostTensor(outputHost.get());
    memcpy(outputBuffer, outputHost->host<float>(), outputSizeByBytes);
    
    return 0;
}

std::vector<int> MnnEngine::getInputShape() {
    auto input = interpreter_->getSessionInput(session_, nullptr);
    return input->shape();
}

std::vector<int> MnnEngine::getOutputShape() {
    auto output = interpreter_->getSessionOutput(session_, NULL);
    return output->shape();
}

const char* MnnEngine::getVersion() {
    return MNN::getVersion();
}

