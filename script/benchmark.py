import os
import sys
import json

executeFile = "../bin/benchmark.out"
libraryFiles = ["../library/libs/tflite/arm64-v8a/libtensorflowlite_jni.so",
               "../library/libs/tflite/arm64-v8a/libtensorflowlite_c.so",
               "../library/libs/tflite/arm64-v8a/libtensorflowlite_gpu_jni.so",
               "../library/libs/MNN/arm64-v8a/libMNN.so",
               "../library/libs/MNN/arm64-v8a/libMNN_CL.so",
               "../library/libs/mindspore/arm64-v8a/libmindspore-lite.so",]
workDir = "/data/local/tmp/benchmark/"

def run(modelInfo):
    with open(modelInfo, 'r') as file:
        infos = json.load(file)
        for info in infos:
            engine = info['engine']

            modelFile = info['modelFile']
            modelFileInDevice = workDir + getFileName(modelFile)
            pushFileToDevice(modelFile, modelFileInDevice)

            inputShape = ""
            if 'inputShape' in info.keys() and info['inputShape']:
                inputShape = "\"" + ",".join([str(x) for x in info['inputShape']]) + "\""

            inputFile = ""
            inputFileInDevice = ""
            if 'inputFile' in info.keys() and info['inputFile']:
                inputFile = info['inputFile']
                inputFileInDevice = workDir + getFileName(inputFile)
                pushFileToDevice(inputFile, inputFileInDevice)
            
            outputFile = ""
            outputFileInDevice = ""
            if 'outputFile' in info.keys() and info['outputFile']:
                outputFile = info['outputFile']
                outputFileInDevice = workDir + getFileName(outputFile)
                pushFileToDevice(outputFile, outputFileInDevice)

            execFileInDevice = workDir + getFileName(executeFile)
            args = []
            args.append(f"-e {engine}")
            args.append(f"-m {modelFileInDevice}")
            args.append(f"-r {inputShape}" if inputShape else "")
            args.append(f"-i {inputFileInDevice}" if inputFile else "")
            args.append(f"-o {outputFileInDevice}" if outputFile else "")
            for backend in info['backend']:
                backend = backend.upper()
                args.append(f"-b {backend}")
                if backend == 'CPU':
                    threadsList = [1]
                    if 'threads' in info.keys() and info ['threads']:
                        threadsList = info ['threads']
                    for threads in threadsList:
                        args.append(f"-t {threads}")
                        print(f"\n*** run {getFileName(modelFile)} with {engine} on {backend.upper()} backend, using {threads} threads ***")
                        os.system(f"adb shell \"export LD_LIBRARY_PATH={workDir} && {execFileInDevice} {' '.join(args)}\"")
                        args.pop()
                else:
                    print(f"\n*** run {getFileName(modelFile)} with {engine} on {backend.upper()} backend ***")
                    os.system(f"adb shell \"export LD_LIBRARY_PATH={workDir} && {execFileInDevice} {' '.join(args)}\"")
                args.pop()


def getFileName(path):
    return os.path.split(path)[1]

def createWorkDir(path):
    os.system(f"adb shell mkdir -p {path}")

def deleteWorkDir(path):
    os.system(f"adb shell rm -rf {path}")

def pushFileToDevice(path, dstPath):
    os.system(f"adb push {path} {dstPath}")

def setup():
    # create workdir
    createWorkDir(workDir)

    # push executable
    execFileInDevice = workDir + getFileName(executeFile)
    pushFileToDevice(executeFile, execFileInDevice)
    os.system(f"adb shell chmod +x {execFileInDevice}")

    # push library
    for libraryFile in libraryFiles:
        libraryFileInDevice = workDir + getFileName(libraryFile)
        pushFileToDevice(libraryFile, libraryFileInDevice)

def end():
    # delete workdir
    deleteWorkDir(workDir)

if __name__ == "__main__":
    modelInfo = sys.argv[1]
    setup()
    run(modelInfo)
    end()
