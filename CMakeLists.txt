cmake_minimum_required(VERSION 3.18)
project(cuda_sgemm LANGUAGES CXX CUDA)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

add_executable(sgemm src/main.cu)
target_link_libraries(sgemm PRIVATE CUDA::cublas)
# -lineinfo lets Nsight Compute map SASS back to source lines.
target_compile_options(sgemm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
