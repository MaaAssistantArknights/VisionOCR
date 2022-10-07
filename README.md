# VisionOCR

PaddleOCR compatible API implemented with Apple Vision framework.

## Build

Download and extract pre-built OpenCV framework from [releases](https://github.com/hguandl/VisionOCR/releases).

```bash
# Set framework search path
$ export CMAKE_FRAMEWORK_PATH=/path/to/opencv2.xcframework/macos-arm64_x86_64

# CMake build
$ mkdir build
$ cd build
$ cmake ..
$ cmake --build .
```
