#include <stdio.h>
#include <iostream>
#include <unistd.h>
#include <getopt.h>
#include <string>
#include "Engine.h"
#include "EngineFactory.h"
#include "TimeUtil.h"

#define CHECK_ERROR(ret, tag) \
    if (ret != 0) { printf(#tag " failed.\n"); return 0;}


std::vector<int> convertStringToIntVector(std::string intVectorString) {
    std::vector<int> shape;

    size_t pos = 0;
    while ((pos = intVectorString.find(',')) != std::string::npos) {
        shape.push_back(stoi(intVectorString.substr(0, pos)));
        intVectorString.erase(0, pos + 1);
    }
    shape.push_back(stoi(intVectorString.substr(0, pos)));

    return shape;
}

int getFileSize(const std::string &fileName) {
    FILE *fp = fopen(fileName.c_str(), "rb");  
    if (!fp) return -1;  
    fseek(fp, 0L, SEEK_END);  
    int size = ftell(fp);  
    fclose(fp);  
      
    return size;  
}

int readFileToBuffer(void* buffer, int size, const std::string &fileName) {
    FILE *fp = fopen(fileName.c_str(), "rb");
    if (!fp) return -1;
    fread(buffer, 1, size, fp);
    fclose(fp);
    return 0;
}

int writeFileFromBuffer(const std::string &fileName, void* buffer, int size) {
    FILE *fp = fopen(fileName.c_str(), "wb");
    if (!fp) return -1;
    fwrite(buffer, 1, size, fp);
    fclose(fp);
    return 0;
}

bool compare(float* output, float* expected, int size) {
    float threshold = 0.01;
    for (int i = 0; i < size; i++) {
        float diff = output[i] - expected[i];
        if (diff > threshold || diff < -threshold) {
            printf("Incorrect at index %d, while output[%d] is %f"
                 " but expected %f\n", i, i, output[i], expected[i]);
            return false;
        }
    }
    return true;
}

int main(int argc, char* argv[]) {
    std::string usage = "Usage:\n"\
        "-e: inference engine name.\n"\
        "-m: model file.\n"\
        "-b: backend, CPU by default.\n"\
        "-t: threads, 1 by default.\n"\
        "-r: optional, input shape if you need resize input.\n"\
        "-i: optional, input data file with nhwc format.\n"\
        "-o: optional, output data file with nhwc format.\n";

    // which inference engine
    std::string engineName = "tflite";

    // model buffer
    std::string modelFileName = "";

    // backend your model run on, CPU by default
    std::string backend = "CPU";

    // thread num
    int threads = 1;

    // resized input, optionnal
    std::vector<int> resizedInputShape = {};

    // input and expected output file, empty string by default, which means using zero as input when inference
    std::string inputFileName = "";
    std::string expectedOutputFileName = "";
    bool verify = false;

    int para;
    const char* optString = "e:m:b:t:r:i:o:h";
    while ((para = getopt(argc, argv, optString)) != -1) {
        switch (para) {
            case 'e':
                engineName = optarg;
                break;
            case 'm':
                modelFileName = optarg;
                break;
            case 'b':
                backend = optarg;
                break;
            case 't':
                threads = atoi(optarg);
                break;
            case 'r':
                resizedInputShape = convertStringToIntVector(optarg);
                break;
            case 'i':
                inputFileName = optarg;
                break;
            case 'o':
                expectedOutputFileName = optarg;
                break;
            case 'h':
                printf("%s\n", usage.c_str());
                break;
            case '?':
                printf("Unknown option: %c\n",(char)optopt);
                printf("%s\n", usage.c_str());
                break;
        }
    }

    if (!inputFileName.empty() && !expectedOutputFileName.empty()) {
        verify = true;
    }
    // printf("\n*** run <%s> with %s on %s backend, using %d threads ***\n",
    //     modelFileName.c_str(), engineName.c_str(), backend.c_str(), threads);
    
    int ret = 0;
    const auto engine = EngineFactory::create(engineName);
    printf("%s version: %s.\n", engineName.c_str(), engine->getVersion());

    ENGINE::Options options;
    if (backend == "CPU") {
        options.backend = ENGINE::Backend::cpu;
    } else if (backend == "GPU") {
        options.backend = ENGINE::Backend::gpu;
    }
    options.threads = threads;

    // read model into buffer and init engine
    int modelSize = getFileSize(modelFileName);
    unsigned char* modelBuffer = new unsigned char[modelSize];

    readFileToBuffer(modelBuffer, modelSize, modelFileName);

    TIC(init);
    CHECK_ERROR(engine->init(modelBuffer, modelSize, options), init);
    TOC(init, 1);

    if (!resizedInputShape.empty()) {
        CHECK_ERROR(engine->resizeInput(0, resizedInputShape), "resizeInput");
        CHECK_ERROR(engine->resizeEngine(), "resizeEngine");
    }

    // get input shape
    const std::vector<int> inputShape = engine->getInputShape();
    int inputSize = 1;
    printf("input shape:");
    for (auto &dim : inputShape) {
        inputSize *= dim;
        printf(" %d", dim);
    }
    printf(".\n");

    float* inputBuffer = new float[inputSize];
    if (verify) {
        int inputFileSize = getFileSize(inputFileName);
        if (inputFileSize != (inputSize * sizeof(float))) {
            printf("Size of input file not matched.\n");
            return 0;
        }
        readFileToBuffer(inputBuffer, inputFileSize, inputFileName);
    } else {
        for (int i = 0; i < inputSize; i++) {
            inputBuffer[i] = 0.f;
        }
    }

    CHECK_ERROR(engine->setInput(inputBuffer), "setInput");
    CHECK_ERROR(engine->inference(), "inference");

    // create output buffer
    std::vector<int> outputShape = engine->getOutputShape();
    int outputSize = 1;
    printf("output shape:");
    for (auto &dim : outputShape) {
        outputSize *= dim;
        printf(" %d", dim);
    }
    printf(".\n");

    float* outputBuffer = new float[outputSize];
    CHECK_ERROR(engine->getOutput(outputBuffer), "getOutput");

    if (verify) {
        // read expected output
        float* expectedOutputBuffer = new float[outputSize];
        int expectedOutputFileSize = getFileSize(expectedOutputFileName);
        if (expectedOutputFileSize != (outputSize * sizeof(float))) {
            printf("Size of input file not matched.\n");
            return 0;
        }
        readFileToBuffer(expectedOutputBuffer, expectedOutputFileSize, expectedOutputFileName);
        
        bool pass = compare(outputBuffer, expectedOutputBuffer, outputSize);
        if (!pass) {
            printf("%s: result validation FAILED!\n", engineName.c_str());
            return 0;
        } else {
            printf("%s: result validation PASSED!\n", engineName.c_str());
        }
    }

    // warm up
    int warmUpTimes = 10;
    for (int i = 0; i < warmUpTimes; i++) {
        engine->setInput(inputBuffer);
        engine->inference();
        engine->getOutput(outputBuffer);
    }

    int execTimes = 50;
    TIC(inference);
    for (int i = 0; i < execTimes; i++) {
        engine->setInput(inputBuffer);
        engine->inference();
        engine->getOutput(outputBuffer);
    }
    TOC(inference, execTimes);

    if (modelBuffer) delete[] modelBuffer;
    if (inputBuffer) delete[] inputBuffer;
    if (inputBuffer) delete[] outputBuffer;
    return 0;
}