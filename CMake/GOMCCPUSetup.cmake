#EnsemblePreprocessor defines NVT = 1, GEMC = 2, GCMC = 3, NPT = 4

#NVT (Canonical) Ensemble
set(NVT_flags "-DENSEMBLE=1")
set(NVT_name "GOMC_CPU_NVT")

#Gibbs Ensemble Monte Carlo
set(GE_flags "-DENSEMBLE=2")
set(GE_name "GOMC_CPU_GEMC")

#Grand Canonical Monte Carlo
set(GC_flags "-DENSEMBLE=3")
set(GC_name "GOMC_CPU_GCMC")

#NPT (Isothermal-Isobaric) Ensemble
set(NPT_flags "-DENSEMBLE=4")
set(NPT_name "GOMC_CPU_NPT")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED true)

# --- FFTW3 detection ---
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(FFTW3 QUIET fftw3)
endif()
if(NOT FFTW3_FOUND)
  find_library(FFTW3_LIBRARIES NAMES fftw3 libfftw3)
  find_path(FFTW3_INCLUDE_DIRS NAMES fftw3.h)
  if(FFTW3_LIBRARIES)
    set(FFTW3_FOUND TRUE)
  endif()
endif()
if(NOT FFTW3_FOUND)
  message(FATAL_ERROR "FFTW3 not found. Install libfftw3-dev or set FFTW3_LIBRARIES manually.")
endif()
message(STATUS "FFTW3 libraries: ${FFTW3_LIBRARIES}")

if(ENSEMBLE_NVT)
   add_executable(NVT ${sources} ${headers} ${libHeaders} ${libSources})
   # Set compiler and linker flags for each compiler
    target_compile_options(NVT
       PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${CMAKE_COMP_FLAGS}>)
    target_link_options(NVT
       PUBLIC $<$<LINK_LANGUAGE:CXX>:${CMAKE_LINK_FLAGS}>)
   set_target_properties(NVT PROPERTIES 
      OUTPUT_NAME ${NVT_name}
      COMPILE_FLAGS "${NVT_flags}")
   target_include_directories(NVT PRIVATE ${FFTW3_INCLUDE_DIRS})
   target_link_libraries(NVT ${FFTW3_LIBRARIES})
   if(WIN32)
      #needed for hostname
      target_link_libraries(NVT ws2_32)
   endif()
   if(MPI_FOUND)
      target_link_libraries(NVT ${MPI_LIBRARIES})
   endif()
endif()

if(ENSEMBLE_GEMC)
   add_executable(GEMC ${sources} ${headers} ${libHeaders} ${libSources})
   # Set compiler and linker flags for each compiler
    target_compile_options(GEMC
       PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${CMAKE_COMP_FLAGS}>)
    target_link_options(GEMC
       PUBLIC $<$<LINK_LANGUAGE:CXX>:${CMAKE_LINK_FLAGS}>)
   set_target_properties(GEMC PROPERTIES 
      OUTPUT_NAME ${GE_name}
      COMPILE_FLAGS "${GE_flags}")
   target_include_directories(GEMC PRIVATE ${FFTW3_INCLUDE_DIRS})
   target_link_libraries(GEMC ${FFTW3_LIBRARIES})
   if(WIN32)
      #needed for hostname
      target_link_libraries(GEMC ws2_32)
   endif()
   if(MPI_FOUND)
      target_link_libraries(GEMC ${MPI_LIBRARIES})
   endif()
endif()

if(ENSEMBLE_GCMC)
   add_executable(GCMC ${sources} ${headers} ${libHeaders} ${libSources})
   # Set compiler and linker flags for each compiler
    target_compile_options(GCMC
       PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${CMAKE_COMP_FLAGS}>)
    target_link_options(GCMC
       PUBLIC $<$<LINK_LANGUAGE:CXX>:${CMAKE_LINK_FLAGS}>)
   set_target_properties(GCMC PROPERTIES 
      OUTPUT_NAME ${GC_name}
      COMPILE_FLAGS "${GC_flags}")
   target_include_directories(GCMC PRIVATE ${FFTW3_INCLUDE_DIRS})
   target_link_libraries(GCMC ${FFTW3_LIBRARIES})
   if(WIN32)
      #needed for hostname
      target_link_libraries(GCMC ws2_32)
   endif()
   if(MPI_FOUND)
      target_link_libraries(GCMC ${MPI_LIBRARIES})
   endif()
endif()

if(ENSEMBLE_NPT)
   add_executable(NPT ${sources} ${headers} ${libHeaders} ${libSources})
   # Set compiler and linker flags for each compiler
    target_compile_options(NPT
       PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${CMAKE_COMP_FLAGS}>)
    target_link_options(NPT
       PUBLIC $<$<LINK_LANGUAGE:CXX>:${CMAKE_LINK_FLAGS}>)
   set_target_properties(NPT PROPERTIES 
      OUTPUT_NAME ${NPT_name}
      COMPILE_FLAGS "${NPT_flags}")
   target_include_directories(NPT PRIVATE ${FFTW3_INCLUDE_DIRS})
   target_link_libraries(NPT ${FFTW3_LIBRARIES})
   if(WIN32)
      #needed for hostname
      target_link_libraries(NPT ws2_32)
   endif()
   if(MPI_FOUND)
      target_link_libraries(NPT ${MPI_LIBRARIES})
   endif()
endif()
