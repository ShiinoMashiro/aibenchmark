cd ..
if exist build (
    rmdir /S /Q build
)
mkdir build & cd build

cmake .. ^
    -DCMAKE_SYSTEM_NAME=Android ^
    -DCMAKE_ANDROID_NDK="C:\Users\xijun.chen\AppData\Local\Android\Sdk\ndk\21.4.7075529" ^
    -DCMAKE_TOOLCHAIN_FILE="C:\Users\xijun.chen\AppData\Local\Android\Sdk\ndk\21.4.7075529/build/cmake/android.toolchain.cmake" ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DANDROID_ABI="arm64-v8a" ^
    -DANDROID_STL=c++_static ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DANDROID_PLATFORM=android-21  ^
    -DANDROID_TOOLCHAIN=clang ^
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ^
    -GNinja

cmake --build .
copy benchmark.out ..\bin\benchmark.out