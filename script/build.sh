#!/bin/bash

cd ..
if [ -d "build" ]; then
  rm -rf build
fi
mkdir build && cd build

ANDROID_NDK=/home/chenxj/env/android-ndk-r21e

cmake .. \
    -DCMAKE_SYSTEM_NAME=Android \
    -DCMAKE_ANDROID_NDK=${ANDROID_NDK} \
    -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK}/build/cmake/android.toolchain.cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_STL=c++_static \
    -DANDROID_PLATFORM=android-21  \
    -DANDROID_TOOLCHAIN=clang \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

cmake --build .
mv benchmark.out ../bin/benchmark.out