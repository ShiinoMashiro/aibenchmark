#include "TfliteEngine.h"

int TfliteEngine::init(const void* modelBuffer, int modelSize, ENGINE::Options options) {
    // Load the model
    const auto model_ = TfLiteModelCreate(modelBuffer, modelSize);
    if (!model_) {
        printf("Failed to create TFLite model.\n");
        return -1;
    }

    // Create the interpreter options
    auto tfliteOptions = TfLiteInterpreterOptionsCreate();

    // Choose CPU or GPU
    if (options.backend == ENGINE::Backend::gpu) {
        delegate_ = TfLiteGpuDelegateV2Create(/*default options=*/nullptr);
        TfLiteInterpreterOptionsAddDelegate(tfliteOptions, delegate_);
    } else {
        // auto xnnpack_options = TfLiteXNNPackDelegateOptionsDefault();
        // xnnpack_options.flags |= TFLITE_XNNPACK_DELEGATE_FLAG_FORCE_FP16;
        // delegate_ = TfLiteXNNPackDelegateCreate(&xnnpack_options);
        // TfLiteInterpreterOptionsAddDelegate(tfliteOptions, delegate_);
        TfLiteInterpreterOptionsSetNumThreads(tfliteOptions, options.threads);
    }

    // Create the interpreter
    interpreter_ = TfLiteInterpreterCreate(model_, tfliteOptions);
    if (!interpreter_) {
        printf("Failed to create TFLite interpreter.\n");
        return -1;
    }

    // Allocate tensors and populate the input tensor data
    TfLiteStatus status = TfLiteInterpreterAllocateTensors(interpreter_);
    if (status != kTfLiteOk) {
        printf("Something went wrong when allocating tensors.\n");
        return -1;
    }

    return 0;
}

int TfliteEngine::resizeInput(int inputIndex, const std::vector<int> &inputShape) {
    // Resizes the specified input tensor.
    TfLiteStatus status = TfLiteInterpreterResizeInputTensor(interpreter_, inputIndex, 
    inputShape.data(), inputShape.size());
    if (status != kTfLiteOk) {
        printf("Something went wrong when resize input.\n");
        return -2;
    }
    return 0;
}

int TfliteEngine::resizeEngine() {
    // Allocate tensors and populate the input tensor data
    TfLiteStatus status = TfLiteInterpreterAllocateTensors(interpreter_);
    if (status != kTfLiteOk) {
        printf("Something went wrong when allocating tensors.\n");
        return -3;
    }

    return 0;
}

int TfliteEngine::setInput(const float* inputBuffer) {
    TfLiteTensor* inputTensor =
        TfLiteInterpreterGetInputTensor(interpreter_, 0);
    const int inputSizeByBytes = inputTensor->bytes;

    // Feed input into model
    auto status = TfLiteTensorCopyFromBuffer(
        inputTensor, inputBuffer, inputSizeByBytes);
    if (status != kTfLiteOk) {
        printf("Something went wrong when copying input buffer to input tensor.\n");
        return -4;
    }

    return 0;
}

int TfliteEngine::inference() {
    // Run the interpreter
    TfLiteStatus status = TfLiteInterpreterInvoke(interpreter_);
    if (status != kTfLiteOk) {
        printf("Something went wrong when running the TFLite model.\n");
        return -5;
    }
    return 0;
}

int TfliteEngine::getOutput(float* outputBuffer) {
    // Extract the output tensor data
    const TfLiteTensor* outputTensor =
        TfLiteInterpreterGetOutputTensor(interpreter_, 0);
    const int outputSizeByBytes = outputTensor->bytes;
                             
    
    auto status = TfLiteTensorCopyToBuffer(
        outputTensor, outputBuffer, outputSizeByBytes);
    if (status != kTfLiteOk) {
        printf("Something went wrong when copying output tensor to output buffer.\n");
        return -6;
    }
    return 0;
}

std::vector<int> TfliteEngine::getInputShape() {
    TfLiteTensor* inputTensor =
        TfLiteInterpreterGetInputTensor(interpreter_, 0);
    int dimsSize = inputTensor->dims->size;
    std::vector<int> res;
    for (int i = 0; i < dimsSize; i++) {
        res.push_back(inputTensor->dims->data[i]);
    }
    return res;
}

std::vector<int> TfliteEngine::getOutputShape() {
    const TfLiteTensor* outputTensor =
        TfLiteInterpreterGetOutputTensor(interpreter_, 0);
    int dimsSize = outputTensor->dims->size;
    std::vector<int> res;
    for (int i = 0; i < dimsSize; i++) {
        res.push_back(outputTensor->dims->data[i]);
    }
    return res;
}

const char* TfliteEngine::getVersion() {
    return TfLiteVersion();
}

