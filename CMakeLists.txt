cmake_minimum_required (VERSION 3.0) 

# This is used for code completion in vim
set(CMAKE_EXPORT_COMPILE_COMMANDS=ON)
project(LASP)

# Whether we want to use blas yes or no
set(LASP_USE_BLAS TRUE)
# set(LASP_USE_BLAS FALSE)

set(LASP_FLOAT double)
# set(LASP_FLOAT float)

add_definitions(-DLASP_PARALLEL)
add_definitions(-DLASP_MAX_NUM_THREADS=8)

add_definitions(-DLASP_MAX_NUM_CHANNELS=80)

# Reasonable maximum to the nfft size, at 48kHz this is 700s of data..
add_definitions(-DLASP_MAX_NFFT=33554432) # 2**25

# ####################################### End of user-adjustable variables section

add_definitions(-D_REENTRANT)

# ############### Choose an fft backend here
#set(LASP_FFT_BACKEND fftpack)
set(LASP_FFT_BACKEND "fftw")

if(LASP_FFT_BACKEND STREQUAL "fftw")
    find_library(FFTW_LIBRARY NAMES fftw3 fftw)
    set(FFTW_LIBRARIES "${FFTW_LIBRARY}")
    set(LASP_FFT_LIBRARY "${FFTW_LIBRARIES}")
    add_definitions(-DLASP_FFT_BACKEND_FFTW)
elseif(LASP_FFT_BACKEND STREQUAL "fftpack")
    add_definitions(-DLASP_FFT_BACKEND_FFTPACK)
    set(LASP_FFT_LIBRARY "fftpack")
endif()


if(LASP_FLOAT STREQUAL "double")
    add_definitions(-DLASP_FLOAT=64)
    add_definitions(-DLASP_DOUBLE_PRECISION)
else()
    add_definitions(-DLASP_FLOAT=32)
    add_definitions(-DLASP_SINGLE_PRECISION)
endif(LASP_FLOAT STREQUAL "double")

if(NOT DEFINED LASP_DEBUG)
    message(FATAL_ERROR "LASP_DEBUG flag not defined. Please set -DLASP_DEBUG=TRUE
    or -DLASP_DEBUG=FALSE")
endif(NOT DEFINED LASP_DEBUG)

# ##################### END Cmake variables converted to a macro
set(Python_ADDITIONAL_VERSIONS "3")
# #################### Setting definitions and debug-specific compilation flags

if(LASP_DEBUG)
    set(TRACERNAME LASPTracer)
    set(LASP_DEBUG_CYTHON=True)
    set(CMAKE_BUILD_TYPE Debug)
    message("Building debug code")
    set(CMAKE_BUILD_TYPE Debug)
    add_definitions(-DLASP_DEBUG=1)
    add_definitions(-DTRACERNAME=${TRACERNAME})
    add_definitions(-DDEBUG)
    add_definitions(-DTRACER=1)
    # This will produce html files
    set(CYTHON_ANNOTATE ON)
    # Add the __FILENAME__ macro
    # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__FILENAME__='\"$(subst ${CMAKE_SOURCE_DIR}/,,$(abspath $<))\"'")
else()
    message("Building LASP for release")
    set(CMAKE_BUILD_TYPE Release)
    set(LASP_DEBUG_CYTHON=False)
    set(CYTHON_ANNOTATE OFF)
    set(CYTHON_NO_DOCSTRINGS ON)
    # Strip unnecessary symbols
    # set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--gc-sections")
    # set(CMAKE_MODULE_LINKER_FLAGS "-Wl,--gc-sections")

    add_definitions(-DTRACER=0 -DNDEBUG)
endif(LASP_DEBUG)

# The last argument here takes care of calling SIGABRT when an integer overflow
# occures.
############################## General compilation flags (independent of debug mode, windows or linux)
set(CYTHON_EXTRA_C_FLAGS "-Wno-sign-compare -Wno-cpp -Wno-implicit-fallthrough -Wno-incompatible-pointer-types -Wno-strict-aliasing")

# General make flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC -std=c11  -Wall -Wextra -Wno-type-limits \
-Werror=implicit-function-declaration -Werror=incompatible-pointer-types \
-Werror=return-type")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -std=c++11  -Wall -Wextra \
-Wno-type-limits")

# Debug make flags
set(CMAKE_C_FLAGS_DEBUG "-g" )

set(CMAKE_C_FLAGS_RELEASE  "-O2 -mfpmath=sse -march=x86-64 -mtune=native \
-fdata-sections -ffunction-sections -fomit-frame-pointer -finline-functions")

# set(CMAKE_C_FLAGS_RELEASE  "-O2 -march=native -mtune=native -fomit-frame-pointer")

if(LASP_USE_BLAS)
    add_definitions(-DLASP_USE_BLAS=1)
else()
    add_definitions(-DLASP_USE_BLAS=0)
endif(LASP_USE_BLAS)

# ############################# End compilation flags

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(win32 true)
else()
    set(win32 false)
endif(CMAKE_SYSTEM_NAME STREQUAL "Windows")

find_package(PythonLibs REQUIRED )
find_package(PythonInterp REQUIRED)

add_subdirectory(fftpack)
include_directories(
    fftpack
    lasp/c
    )

add_subdirectory(lasp)
add_subdirectory(test)

# set(SETUP_PY_IN "${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in")
set(SETUP_PY    "${CMAKE_CURRENT_BINARY_DIR}/setup.py")

set(DEPS  "${CMAKE_CURRENT_SOURCE_DIR}/*.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/lasp/*.py"
    "wrappers"
    "lasp_rtaudio")

# )
set(OUTPUT      "${CMAKE_CURRENT_BINARY_DIR}/build/timestamp")

# configure_file(${SETUP_PY_IN} ${SETUP_PY})
add_custom_command(OUTPUT ${OUTPUT}
    COMMAND ${PYTHON_EXECUTABLE} ${SETUP_PY} build
    COMMAND ${CMAKE_COMMAND} -E touch ${OUTPUT}
    DEPENDS ${DEPS})

add_custom_target(target ALL DEPENDS ${OUTPUT})

if(DEFINED INSTALL_DEBUG)
    set(EXTRA_SETUP_ARG --user -e)
else()
    set(EXTRA_SETUP_ARG "")
endif()


install(CODE "execute_process(COMMAND ${PYTHON_EXECUTABLE} -m pip install ${EXTRA_SETUP_ARG} .)")
