
include(FetchContent)
# Avoid warning about DOWNLOAD_EXTRACT_TIMESTAMP in CMake 3.24:
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
  cmake_policy(SET CMP0135 NEW)
endif()

# download onnxruntime version={1.17.1}
FetchContent_Declare(
  onnxruntime
  URL https://github.com/microsoft/onnxruntime/releases/download/v1.18.1/onnxruntime-linux-x64-gpu-1.18.1.tgz
  URL_HASH SHA256=d947af0e4311fd38012ad69dea4983e73ce5f1754da0d5b7a118603dd87b197d
  DOWNLOAD_NO_EXTRACT false
)

if (NOT onnxruntime_POPULATED)
  FetchContent_Populate(onnxruntime)
endif()

