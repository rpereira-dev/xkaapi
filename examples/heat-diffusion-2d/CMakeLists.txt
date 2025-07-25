# CMake Minimum version
cmake_minimum_required(VERSION 3.7)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)

# Project name
SET(HEAT_NAME heat)
project(${HEAT_NAME})

SET(SOURCE_FILES
    ${CMAKE_SOURCE_DIR}/src/main.cc
)

SET(HEAT_CUDA_KERNELS_SOURCE_FILES
    ${CMAKE_SOURCE_DIR}/src/kernels/diffusion.cu
)

SET(HEAT_HIP_KERNELS_SOURCE_FILES
    ${CMAKE_SOURCE_DIR}/src/kernels/diffusion-hip.cc
)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

option(USE_CUDA "Enable the compilation of cuda kernels" OFF)
option(USE_ZE "Enable the compilation of ze kernels" OFF)
option(USE_SYCL "Enable the compilation of SYCL and set its interop with Level Zero driver" OFF)
option(USE_HIP "Enable the compilation of hip kernels" OFF)

###############
# C++ version #
###############
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)  # Optional: disables compiler-specific extensions

########
# HEAT #
########
add_executable(${HEAT_NAME} ${SOURCE_FILES})

# Include directory
include_directories(include/)
include_directories(src/)

################
# DEPENDENCIES #
################

# XK Runtime
find_package(XKRT REQUIRED)
include_directories("${XKRT_INCLUDE_DIRS}")
target_link_libraries(${HEAT_NAME} PRIVATE ${XKRT_LIBRARIES})

# Check that at least one option is ON
if(NOT (USE_CUDA OR USE_SYCL OR USE_ZE OR USE_HIP))
    message(FATAL_ERROR "At least one of USE_CUDA, USE_SYCL, USE_ZE or USE_HIP must be enabled.")
endif()

# Cuda kernels
if (USE_CUDA)

    find_package(CUDA_DRIVER REQUIRED)
    find_package(CUDA_RT REQUIRED)
    include_directories("${CUDA_DRIVER_INCLUDE_DIRS}")
    include_directories("${CUDA_RT_INCLUDE_DIRS}")
    target_link_libraries(${HEAT_NAME} PRIVATE ${CUDA_DRIVER_LIBRARIES} ${CUDA_RT_LIBRARIES})

    enable_language(CUDA)
    set_source_files_properties(${HEAT_CUDA_KERNELS_SOURCE_FILES} PROPERTIES LANGUAGE CUDA)
    add_library(${HEAT_NAME}_kernels_cuda SHARED ${HEAT_CUDA_KERNELS_SOURCE_FILES})
    set_target_properties(${HEAT_NAME}_kernels_cuda PROPERTIES CUDA_ARCHITECTURES "50;72")
    install(TARGETS ${HEAT_NAME}_kernels_cuda LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
    target_link_libraries(${HEAT_NAME} PRIVATE ${HEAT_NAME}_kernels_cuda)
endif()

# Level Zero
if (USE_ZE)
    find_package(ZE REQUIRED)
    include_directories("${ZE_INCLUDE_DIRS}")
    target_link_libraries(${HEAT_NAME} PRIVATE ${ZE_LIBRARIES})
endif()

# HIP
if (USE_HIP)
    find_package(HIP REQUIRED)
    include_directories("${HIP_INCLUDE_DIRS}")
    target_link_libraries(${HEAT_NAME} PRIVATE ${HIP_LIBRARIES})

    enable_language(HIP)
    set_source_files_properties(${HEAT_HIP_KERNELS_SOURCE_FILES} PROPERTIES LANGUAGE HIP)
    add_library(${HEAT_NAME}_kernels_hip SHARED ${HEAT_HIP_KERNELS_SOURCE_FILES})
    set_target_properties(${HEAT_NAME}_kernels_hip PROPERTIES HIP_ARCHITECTURES "gfx90a")
    install(TARGETS ${HEAT_NAME}_kernels_hip LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
    target_link_libraries(${HEAT_NAME} PRIVATE ${HEAT_NAME}_kernels_hip)
endif()

# SYCL
if (USE_SYCL)
    find_package(SYCL REQUIRED)
    include_directories(${SYCL_INCLUDE_DIRS})
    target_compile_options(${HEAT_NAME} PRIVATE -fsycl)
    target_link_libraries(${HEAT_NAME} PRIVATE ${SYCL_LIBRARIES} -fsycl)
endif()

