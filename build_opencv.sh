#!/bin/sh

if [ ! -f "opencv-4.5.3/CMakeLists.txt" ]; then
    echo "Downloading OpenCV 4.5.3"
    curl -fSL -o- https://github.com/opencv/opencv/archive/4.5.3.tar.gz | tar zxf -
    # Fix mac-catalyst options
    curl -fsSL -o- https://github.com/opencv/opencv/commit/751b3f502da8ed5c2a6346ca58ef12c687b20fd5.diff | patch -p1 -d opencv-4.5.3
fi

echo "Building OpenCV 4.5.3"
pushd opencv-4.5.3
python3 platforms/apple/build_xcframework.py -o ../opencv-build \
    --macos_archs x86_64,arm64 \
    --build_only_specified_archs \
    --macosx_deployment_target 11.0 \
    --disable FFMPEG --disable PROTOBUF \
    --without objc \
    --disable-swift
popd

echo "Build complete. OpenCV xcframework is at opencv-build/opencv2.xcframework"
