#ifndef _TIME_UTIL_H
#define _TIME_UTIL_H

#include <ctime>
#include <chrono>

#define TIC(tag) auto time_##tag##_start = std::chrono::high_resolution_clock::now()
#define TOC(tag, times) auto time_##tag##_end = std::chrono::high_resolution_clock::now();\
        std::chrono::duration<double> time_##tag##_elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(time_##tag##_end - time_##tag##_start);\
        printf(#tag " time: %.3f ms.\n", time_##tag##_elapsed.count() * 1000 / times)

#endif //_TIME_UTIL_H