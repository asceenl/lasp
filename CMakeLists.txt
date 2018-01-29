cmake_minimum_required (VERSION 3.0)
project(beamforming)

# Whether we want to use blas yes or no
set(ASCEE_USE_BLAS 1)
add_definitions(-DASCEE_USE_BLAS=1)

set(ASCEE_FLOAT double)
# set(ASCEE_FLOAT float)

add_definitions(-DASCEE_PARALLEL)
add_definitions(-DASCEE_MAX_NUM_THREADS=8)

add_definitions(-DASCEE_MAX_NUM_CHANNELS=80)

# ####################################### End of user-adjustable variables section

add_definitions(-D_REENTRANT)


if(ASCEE_FLOAT STREQUAL "double")
  add_definitions(-DASCEE_FLOAT=64)
  add_definitions(-DASCEE_DOUBLE_PRECISION)
else()
  add_definitions(-DASCEE_FLOAT=32)
endif(ASCEE_FLOAT STREQUAL "double")


if(NOT DEFINED ASCEE_DEBUG)
  message(SEND_ERROR "ASCEE_DEBUG flag not defined. Please set -DASCEE_DEBUG=TRUE or -DASCEE_DEBUG=FALSE")
endif(NOT DEFINED ASCEE_DEBUG)

# ##################### END Cmake variables converted to a macro
set(Python_ADDITIONAL_VERSIONS "3")
# #################### Setting definitions and debug-specific compilation flags

if(ASCEE_DEBUG)
  set(TRACERNAME ASCEETracer)
  set(CMAKE_BUILD_TYPE Debug)
  message("Building debug code")
  set(CMAKE_BUILD_TYPE Debug)
  add_definitions(-DASCEE_DEBUG=1)
  add_definitions(-DTRACERNAME=${TRACERNAME})
  add_definitions(-DDEBUG)
  add_definitions(-DTRACER=1)
  
  set(CYTHON_VARIABLES "#cython: boundscheck=True")
  # This will produce html files
  set(CYTHON_ANNOTATE ON)
  # Add the __FILENAME__ macro
  # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__FILENAME__='\"$(subst ${CMAKE_SOURCE_DIR}/,,$(abspath $<))\"'")
else()
  message("Building release code")
  set(CMAKE_BUILD_TYPE Release)
  set(CYTHON_ANNOTATE OFF)
  set(CYTHON_NO_DOCSTRINGS ON)
  set(CYTHON_VARIABLES "#cython: boundscheck=False,wraparound=False,embedsignature=False,emit_code_comments=False")
  # Strip unnecessary symbols
  # set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--gc-sections")
  # set(CMAKE_MODULE_LINKER_FLAGS "-Wl,--gc-sections")
  
  add_definitions(-DTRACER=0 -DNDEBUG)
endif(ASCEE_DEBUG)

# The last argument here takes care of calling SIGABRT when an integer overflow
# occures.
############################## General compilation flags (independent of debug mode, windows or linux)

# General make flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC -std=c11  -Wall -Wextra \
  -Wimplicit-function-declaration -ftrapv -Wno-type-limits")

# Debug make flags
set(CMAKE_C_FLAGS_DEBUG "-g" )

set(CMAKE_C_FLAGS_RELEASE  "-O2 -mfpmath=sse -march=x86-64 -mtune=native \
  -fdata-sections -ffunction-sections -fomit-frame-pointer -finline-functions")

# set(CMAKE_C_FLAGS_RELEASE  "-O2 -march=native -mtune=native -fomit-frame-pointer")


if(ASCEE_USE_BLAS)
  add_definitions(-DASCEE_USE_BLAS=1)  
else()
  add_definitions(-DASCEE_USE_BLAS=0)  
endif(ASCEE_USE_BLAS)

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  set(win32 true)
else()
  set(win32 false)
endif(CMAKE_SYSTEM_NAME STREQUAL "Windows")

# If numpy cannot be found in the standard include path of the Python 
if(DEFINED NUMPY_INCLUDE)
  include_directories(${NUMPY_INCLUDE})
endif(DEFINED NUMPY_INCLUDE)




# ############################# End compilation flags

add_subdirectory(fftpack)
include_directories(
  fftpack
  beamforming/c
  )
add_subdirectory(beamforming)
add_subdirectory(test)

find_program(PYTHON "python")

if (PYTHON)
    # set(SETUP_PY_IN "${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in")
    set(SETUP_PY    "${CMAKE_CURRENT_BINARY_DIR}/setup.py")
    set(DEPS        "${CMAKE_CURRENT_SOURCE_DIR}/beamforming/*.py")
    set(OUTPUT      "${CMAKE_CURRENT_BINARY_DIR}/build/timestamp")

    # configure_file(${SETUP_PY_IN} ${SETUP_PY})

    add_custom_command(OUTPUT ${OUTPUT}
                       COMMAND ${PYTHON} ${SETUP_PY} build
                       COMMAND ${CMAKE_COMMAND} -E touch ${OUTPUT}
                       DEPENDS ${DEPS})

    add_custom_target(target ALL DEPENDS ${OUTPUT})

    install(CODE "execute_process(COMMAND ${PYTHON} ${SETUP_PY} install)")
endif()


############################## End compiler settings

