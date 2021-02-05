# Set the minimum required version of CMake for this project.
cmake_minimum_required(VERSION 3.7)

# Set this project's name.
project("dss")

# Set the C++ standard to use to build this target.
set(CMAKE_CXX_STANDARD 11)

# Enable CTest for testing these code examples.
# include(CTest)

set(DSS_LIB "dss")
set(DSS_LIB_TEST "test_lib")

# Build shared libraries by default.
if(NOT BUILD_SHARED_LIBS)
    set(BUILD_SHARED_LIBS ON)
endif()

# Find the AWS SDK for C++ package.
find_package(AWSSDK REQUIRED COMPONENTS s3)

SET(CMAKE_CXX_FLAGS "-O0 -g3 -Wall -Werror")

set(SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/dss_client.cpp)

set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

add_executable(test_dss ${SOURCES})
add_library(${DSS_LIB} SHARED ${SOURCES})

target_compile_definitions(${DSS_LIB} PUBLIC "DSS_DEBUG")
target_compile_definitions(test_dss PUBLIC "DSS_DEBUG")

target_include_directories(test_dss PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_include_directories(${DSS_LIB} PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

target_link_libraries(test_dss ${AWSSDK_LINK_LIBRARIES})
target_link_libraries(${DSS_LIB} ${AWSSDK_LINK_LIBRARIES})

execute_process(COMMAND python3-config --extension-suffix OUTPUT_VARIABLE EXT_NAME_SUFFIX)
execute_process(COMMAND python3 -m pybind11 --includes OUTPUT_VARIABLE EXT_INCLUDE_FLAG)
separate_arguments(EXT_INCLUDE_FLAGS UNIX_COMMAND "${EXT_INCLUDE_FLAG}")
message("${EXT_INCLUDE_FLAGS}")

string(REGEX REPLACE "\\.[^.]*$" "" PYTHON_LIB_SUFFIX ${EXT_NAME_SUFFIX})
string(CONCAT DSS_PYTHON_LIB ${DSS_LIB} ${PYTHON_LIB_SUFFIX})

add_library(${DSS_PYTHON_LIB} SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/dss_ext.cpp)
target_include_directories(${DSS_PYTHON_LIB}  PUBLIC
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_compile_options(${DSS_PYTHON_LIB} PRIVATE ${EXT_INCLUDE_FLAGS})
target_link_options(${DSS_PYTHON_LIB} PUBLIC "-Wl,--disable-new-dtags" "-Wl,-rpath,/usr/lib/x86_64-linux-gnu/")

target_link_libraries(${DSS_PYTHON_LIB} ${DSS_LIB})
set_target_properties(${DSS_PYTHON_LIB} PROPERTIES PREFIX "")

#g++ -O0 -g3 -Wall -Werror -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` -I./ dss_extension.cpp -o dss`python3.8-config --extension-suffix` -L./build -lDSSClient  -Wl,-rpath,. -Wl,--disable-new-dtags -Wl,-rpath,/usr/lib/x86_64-linux-gnu/
