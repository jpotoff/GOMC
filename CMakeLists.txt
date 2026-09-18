cmake_minimum_required(VERSION 3.18)

project(GOMC)

include_directories(lib)
include_directories(src)
include_directories(src/cbmc)
include_directories(src/moves)
include_directories(src/GPU)

# Versioning
set(GOMC_VERSION_MAJOR 2)
set(GOMC_VERSION_MINOR 80)

# Out-of-source build
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Run clang-tidy
if(GOMC_TIDY)
   set(CMAKE_CXX_CLANG_TIDY clang-tidy -checks=-*,mpi-*,openmp-*)
endif()

if(NOT CMAKE_BUILD_TYPE)
   set(CMAKE_BUILD_TYPE Release CACHE STRING
      "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
      FORCE)
endif(NOT CMAKE_BUILD_TYPE)

# Set compile and link flags. Need to do it this way so that we can pass
# the source compiler flags to NVCC properly
set(CMAKE_INTEL_COMP_FLAGS -O3 -fp-model=precise)
set(CMAKE_INTEL_CUDA_COMP_FLAGS "SHELL:-Xcompiler -O3" "SHELL:-fp-model=precise")
set(CMAKE_INTEL_LINK_FLAGS -O3 -fp-model=precise)
set(CMAKE_GNU_COMP_FLAGS -O3 -m64)
set(CMAKE_GNU_CUDA_COMP_FLAGS "SHELL:-Xcompiler -O3" "SHELL:-Xcompiler -m64")
set(CMAKE_GNU_LINK_FLAGS -m64)
set(CMAKE_CLANG_COMP_FLAGS -O3)
set(CMAKE_CLANG_CUDA_COMP_FLAGS "SHELL:-Xcompiler -O3")
set(CMAKE_CLANG_LINK_FLAGS -O3)

# The GOMC_OPT setting enables processor-specific optimizations.
#
# GOMC_ARCH picks the target. The default, x86-64-v3, is AVX2 + FMA: it runs on
# anything Haswell or Zen 1 and newer, and gives the vector units 4 doubles at a
# time instead of the 2 the baseline x86-64 target allows. It stays the default
# because GOMC is run on a variety of machines and a v3 binary runs on all of
# them.
#
# For benchmarking, prefer a target that enables AVX-512:
#
#   cmake .. -DGOMC_OPT=on -DGOMC_ARCH=znver5      # Zen 4/5, Ryzen 9000
#   cmake .. -DGOMC_OPT=on -DGOMC_ARCH=x86-64-v4   # any AVX-512 part
#
# This matters more than it used to. The minimum-image/distance pass of
# CalculateEnergy::BoxInterTemplate now vectorises (see DistSqRange in
# BoxDimensions.h), and it is 80-92% of the pair work, so the width of the
# vector unit is on the critical path: v3 gives it 4 doubles per iteration and
# AVX-512 gives it 8, with mask registers for the minimum-image selects.
# Before that loop vectorised, x86-64-v3 was worth roughly 6% on the OPC GEMC
# benchmark and AVX-512 added nothing beyond it, because the only vectorised
# code was in reciprocal space.
#
# Set -DGOMC_ARCH=native to tune for the build machine (faster to type, but the
# binary then only runs on that microarchitecture), or to any other -march=
# value such as znver4.
#
# NOTE: do not use Intel's -xHost here. It makes icc/icpx emit a GenuineIntel
# dispatch check, and the binary refuses to start on AMD with "This executable
# is built with target processor feature specified, it can only be run on
# Intel(R) processors". -march= carries no such check. Classic icc predates
# -march= and has no alternative, so it keeps -xHost and stays Intel-only.
if(GOMC_OPT)
   set(GOMC_ARCH "x86-64-v3" CACHE STRING
       "Target architecture for GOMC_OPT (-march=); e.g. x86-64-v3, x86-64-v4, znver5, native")

   if("${CMAKE_CXX_COMPILER_ID}" MATCHES "^Intel$")
      # Classic icc: no -march=, fall back to -xHost (Intel CPUs only).
      set(GOMC_ARCH_FLAG -xHost)
   else()
      set(GOMC_ARCH_FLAG -march=${GOMC_ARCH})
   endif()

   # Older compilers may not know the chosen target; find out before we commit
   # to it rather than failing the whole build over an optional optimisation.
   include(CheckCXXCompilerFlag)
   check_cxx_compiler_flag("${GOMC_ARCH_FLAG}" GOMC_ARCH_FLAG_SUPPORTED)
   if(NOT GOMC_ARCH_FLAG_SUPPORTED)
      message(WARNING "Compiler does not accept ${GOMC_ARCH_FLAG}; "
                      "building without processor-specific optimizations. "
                      "Try -DGOMC_ARCH=native.")
      set(GOMC_ARCH_FLAG "")
   endif()

   if(GOMC_ARCH_FLAG)
      message(STATUS "GOMC_OPT: using ${GOMC_ARCH_FLAG}")
      set(CMAKE_INTEL_COMP_FLAGS ${CMAKE_INTEL_COMP_FLAGS} ${GOMC_ARCH_FLAG})
      set(CMAKE_INTEL_CUDA_COMP_FLAGS ${CMAKE_INTEL_CUDA_COMP_FLAGS} "SHELL:-Xcompiler ${GOMC_ARCH_FLAG}")
      set(CMAKE_INTEL_LINK_FLAGS ${CMAKE_INTEL_LINK_FLAGS} ${GOMC_ARCH_FLAG})
      set(CMAKE_GNU_COMP_FLAGS ${CMAKE_GNU_COMP_FLAGS} ${GOMC_ARCH_FLAG})
      set(CMAKE_GNU_CUDA_COMP_FLAGS ${CMAKE_GNU_CUDA_COMP_FLAGS} "SHELL:-Xcompiler ${GOMC_ARCH_FLAG}")
      set(CMAKE_GNU_LINK_FLAGS ${CMAKE_GNU_LINK_FLAGS} ${GOMC_ARCH_FLAG})
      set(CMAKE_CLANG_COMP_FLAGS ${CMAKE_CLANG_COMP_FLAGS} ${GOMC_ARCH_FLAG})
      set(CMAKE_CLANG_CUDA_COMP_FLAGS ${CMAKE_CLANG_CUDA_COMP_FLAGS} "SHELL:-Xcompiler ${GOMC_ARCH_FLAG}")
      set(CMAKE_CLANG_LINK_FLAGS ${CMAKE_CLANG_LINK_FLAGS} ${GOMC_ARCH_FLAG})
   endif()
endif()


# Set the necessary flags for using address sanitizer, which doesn't work with the Intel compiler
if(GOMC_ASAN)
   set(CMAKE_GNU_COMP_FLAGS ${CMAKE_GNU_COMP_FLAGS} -fsanitize=address -fno-omit-frame-pointer)
   set(CMAKE_GNU_CUDA_COMP_FLAGS ${CMAKE_GNU_CUDA_COMP_FLAGS} "SHELL:-Xcompiler -fsanitize=address" "SHELL:-Xcompiler -fno-omit-frame-pointer")
   set(CMAKE_GNU_LINK_FLAGS ${CMAKE_GNU_LINK_FLAGS} -fsanitize=address -fno-omit-frame-pointer)
   set(CMAKE_CLANG_COMP_FLAGS ${CMAKE_CLANG_COMP_FLAGS} -fsanitize=address -fno-omit-frame-pointer)
   set(CMAKE_CLANG_CUDA_COMP_FLAGS ${CMAKE_CLANG_CUDA_COMP_FLAGS} "SHELL:-Xcompiler -fsanitize=address" "SHELL:-Xcompiler -fno-omit-frame-pointer")
   set(CMAKE_CLANG_LINK_FLAGS ${CMAKE_CLANG_LINK_FLAGS} -fsanitize=address -fno-omit-frame-pointer)
   set(CMAKE_MVSC_ASAN_FLAGS "/fsanitize=address")
endif()

# Find OpenMP and set it up
# Don't use OpenMP if using address sanitizer
if(NOT GOMC_ASAN)
   find_package(OpenMP)
   if(OPENMP_FOUND)
      set(CMAKE_INTEL_COMP_FLAGS ${CMAKE_INTEL_COMP_FLAGS} ${OpenMP_CXX_FLAGS})
      set(CMAKE_INTEL_CUDA_COMP_FLAGS ${CMAKE_INTEL_CUDA_COMP_FLAGS} "SHELL:-Xcompiler ${OpenMP_CXX_FLAGS}")
      set(CMAKE_INTEL_LINK_FLAGS ${CMAKE_INTEL_LINK_FLAGS} ${OpenMP_CXX_FLAGS})
      set(CMAKE_GNU_COMP_FLAGS ${CMAKE_GNU_COMP_FLAGS} ${OpenMP_CXX_FLAGS})
      set(CMAKE_GNU_CUDA_COMP_FLAGS ${CMAKE_GNU_CUDA_COMP_FLAGS} "SHELL:-Xcompiler ${OpenMP_CXX_FLAGS}")
      set(CMAKE_GNU_LINK_FLAGS ${CMAKE_GNU_LINK_FLAGS} ${OpenMP_CXX_FLAGS})
      set(CMAKE_CLANG_COMP_FLAGS ${CMAKE_CLANG_COMP_FLAGS} ${OpenMP_CXX_FLAGS})
      set(CMAKE_CLANG_CUDA_COMP_FLAGS ${CMAKE_CLANG_CUDA_COMP_FLAGS} "SHELL:-Xcompiler ${OpenMP_CXX_FLAGS}")
      set(CMAKE_CLANG_LINK_FLAGS ${CMAKE_CLANG_LINK_FLAGS} ${OpenMP_CXX_FLAGS})
      set(CMAKE_MSVC_OPENMP_FLAGS "/openmp:llvm")
   endif()
endif()

set(ENSEMBLE_NVT ON CACHE BOOL "Build NVT version")
set(ENSEMBLE_GEMC ON CACHE BOOL "Build GEMC version")
set(ENSEMBLE_GCMC ON CACHE BOOL "Build GCMC version")
set(ENSEMBLE_NPT ON CACHE BOOL "Build NPT version")
set(ENSEMBLE_GPU_NVT ON CACHE BOOL "Build GPU NVT version")
set(ENSEMBLE_GPU_GEMC ON CACHE BOOL "Build GPU GEMC version")
set(ENSEMBLE_GPU_GCMC ON CACHE BOOL "Build GPU GCMC version")
set(ENSEMBLE_GPU_NPT ON CACHE BOOL "Build GPU NPT version")

include(${PROJECT_SOURCE_DIR}/CMake/GOMCMPI.cmake)

include_directories("${PROJECT_BINARY_DIR}")

# Set compiler flags now that all the options have been specified
if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Intel" OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "IntelLLVM")
   set(CMAKE_COMP_FLAGS ${CMAKE_INTEL_COMP_FLAGS})
   set(CMAKE_CUDA_COMP_FLAGS ${CMAKE_INTEL_CUDA_COMP_FLAGS} -allow-unsupported-compiler)
   set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler)
   set(CMAKE_LINK_FLAGS ${CMAKE_INTEL_LINK_FLAGS})
elseif("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
   set(CMAKE_COMP_FLAGS ${CMAKE_GNU_COMP_FLAGS})
   set(CMAKE_CUDA_COMP_FLAGS ${CMAKE_GNU_CUDA_COMP_FLAGS})
   set(CMAKE_LINK_FLAGS ${CMAKE_GNU_LINK_FLAGS})
elseif("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
   set(CMAKE_COMP_FLAGS ${CMAKE_CLANG_COMP_FLAGS})
   set(CMAKE_CUDA_COMP_FLAGS ${CMAKE_CLANG_CUDA_COMP_FLAGS})
   set(CMAKE_LINK_FLAGS ${CMAKE_CLANG_LINK_FLAGS})
elseif("${CMAKE_CXX_COMPILER_ID}" MATCHES "MSVC")
   set(CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG} ${CMAKE_MSVC_OPENMP_FLAGS} ${CMAKE_MVSC_ASAN_FLAGS} /D DEBUG /MTd /Zi /Ob0 /Od /RTC1 /D_USE_MATH_DEFINES)
   set(CMAKE_CXX_FLAGS_DEBUG_INIT ${CMAKE_CXX_FLAGS_DEBUG_INIT} ${CMAKE_MSVC_OPENMP_FLAGS} ${CMAKE_MVSC_ASAN_FLAGS} /D DEBUG /MTd /Zi /Ob0 /Od /RTC1 /D_USE_MATH_DEFINES)
   set(CMAKE_CXX_FLAGS_MINSIZEREL_INIT ${CMAKE_CXX_FLAGS_MINSIZEREL_INIT} ${CMAKE_MSVC_OPENMP_FLAGS} /MT /O1 /Ob1 /D NDEBUG /D_USE_MATH_DEFINES)
   set(CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_MINSIZEREL_INIT} ${CMAKE_MSVC_OPENMP_FLAGS} /MT /O2 /D NDEBUG /D_USE_MATH_DEFINES)
   set(CMAKE_CXX_FLAGS_RELEASE_INIT ${CMAKE_CXX_FLAGS_RELEASE_INIT} ${CMAKE_MSVC_OPENMP_FLAGS} /MT /O2 /D NDEBUG /D_USE_MATH_DEFINES)
   set(CMAKE_CXX_FLAGS_RELWITHDEBINFO_INIT ${CMAKE_CXX_FLAGS_RELWITHDEBINFO_INIT} ${CMAKE_MSVC_OPENMP_FLAGS} /MT /Zi /O2 /Ob1 /D NDEBUG /D_USE_MATH_DEFINES)
endif()

# Set Source and Header files
include(${PROJECT_SOURCE_DIR}/CMake/FileLists.cmake)

# Optional CMake Arguments
option(GOMC_GTEST    "Build unit tests for GOMC" OFF)
option(GOMC_GTEST_MPI    "Build unit tests for GOMC - MPI Enabled" OFF)

# Enable config header
configure_file(
    "${PROJECT_SOURCE_DIR}/GOMC_Config.h.in"
    "${PROJECT_BINARY_DIR}/GOMC_Config.h"
)

# Enable google test
# for now we will disable testing for Intel compiler
if(GOMC_GTEST OR GOMC_GTEST_MPI)
  enable_testing()
  include(test/GoogleTest.cmake)
endif()

# Check if CUDA exists and, if so, set the host compiler and enable CUDA
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
   set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
   enable_language(CUDA)
   include(${PROJECT_SOURCE_DIR}/CMake/GOMCCUDASetup.cmake)
else()
   message(WARNING "No CUDA support for this compiler")
endif()

# Set up Serial version
include(${PROJECT_SOURCE_DIR}/CMake/GOMCCPUSetup.cmake)
