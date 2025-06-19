cmake_minimum_required(VERSION 3.10)

# Explicitly set the CUDA compiler
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")

# Set the project name
project(CPP_Benchmark LANGUAGES CXX)

# Use the local LibTorch installation
set(CMAKE_PREFIX_PATH "/home/cs-department-skytechgaming1/Downloads/mnist_linux-master (bad copy)/cpp/libtorch")

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find packages
find_package(Torch REQUIRED)
find_package(CURL REQUIRED)
find_package(CUDA) # Find CUDA, but don't make it required globally

if(CUDA_FOUND)
    message(STATUS "CUDA found, enabling CUDA support.")
    enable_language(CUDA)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
else()
    message(STATUS "CUDA not found, building for CPU only.")
endif()

# Find threads package
find_package(Threads REQUIRED)

# Add executable
add_executable(mnist_benchmark src/main.cpp)

# Link libraries
target_link_libraries(mnist_benchmark PUBLIC "${TORCH_LIBRARIES}")
target_link_libraries(mnist_benchmark PRIVATE CURL::libcurl)
target_link_libraries(mnist_benchmark PRIVATE Threads::Threads)

if(CUDA_FOUND)
    target_link_libraries(mnist_benchmark PRIVATE CUDA::cudart)
endif()

# Set properties for C++ standard if not using target_compile_features
set_property(TARGET mnist_benchmark PROPERTY CXX_STANDARD 17)
set_property(TARGET mnist_benchmark PROPERTY CXX_STANDARD_REQUIRED ON)

if(CUDA_FOUND AND TARGET mnist_benchmark)
    set_property(TARGET mnist_benchmark PROPERTY CUDA_STANDARD 11)
    set_property(TARGET mnist_benchmark PROPERTY CUDA_STANDARD_REQUIRED ON)
endif()
