cmake_minimum_required(VERSION 3.4.1)

project(benchmark)

set(TFLITE_LIBPATH "${CMAKE_CURRENT_SOURCE_DIR}/library/libs/tflite")
set(MNN_LIBPATH "${CMAKE_CURRENT_SOURCE_DIR}/library/libs/MNN")
set(MINDSPORE_LIBPATH "${CMAKE_CURRENT_SOURCE_DIR}/library/libs/mindspore")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++14")
set(CMAKE_CXX_STANDARD 14)

set(TFLITE core/tflite/TfliteEngine.cpp)
set(MNN core/mnn/MnnEngine.cpp)
set(MINDSPORE core/mindspore/MindSporeEngine.cpp)
set(UTILS core/utils/TimeUtil.h)
set(BENCHMARK core/benchmark.cpp)

add_executable(benchmark.out ${TFLITE} ${MNN} ${MINDSPORE} ${UTILS} ${BENCHMARK})

add_library(lib_tensorflowlite SHARED IMPORTED)
set_target_properties(lib_tensorflowlite PROPERTIES IMPORTED_LOCATION
        ${TFLITE_LIBPATH}/${ANDROID_ABI}/libtensorflowlite_c.so)

add_library(lib_tensorflowlite_gpu SHARED IMPORTED)
set_target_properties(lib_tensorflowlite_gpu PROPERTIES IMPORTED_LOCATION
        ${TFLITE_LIBPATH}/${ANDROID_ABI}/libtensorflowlite_gpu_jni.so)

add_library(lib_mnn SHARED IMPORTED)
set_target_properties(lib_mnn PROPERTIES IMPORTED_LOCATION
        ${MNN_LIBPATH}/${ANDROID_ABI}/libMNN.so)

add_library(lib_mnn_gpu SHARED IMPORTED)
set_target_properties(lib_mnn_gpu PROPERTIES IMPORTED_LOCATION
        ${MNN_LIBPATH}/${ANDROID_ABI}/libMNN_CL.so)

add_library(lib_mindspore SHARED IMPORTED)
set_target_properties(lib_mindspore PROPERTIES IMPORTED_LOCATION
        ${MINDSPORE_LIBPATH}/${ANDROID_ABI}/libmindspore-lite.so)

find_library(log-lib log)
find_library(egl-lib EGL)
find_library(gles-lib GLESv3)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/library/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/core/engine)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/core/tflite)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/core/mnn)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/core/mindspore)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/core/utils)

target_link_libraries(benchmark.out
                      lib_tensorflowlite
                      lib_tensorflowlite_gpu
                      lib_mnn
                      lib_mnn_gpu
                      lib_mindspore
                      # Links the target library to the log library
                      # included in the NDK.
                      ${log-lib}
                      ${egl-lib}
                      ${gles-lib})