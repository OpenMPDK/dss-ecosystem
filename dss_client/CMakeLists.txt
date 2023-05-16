# Set the minimum required version of CMake for this project.
cmake_minimum_required(VERSION 3.7)

# Set this project's name.
project("dss")

# Set the C++ standard to use to build this target.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED on)

#set(CMAKE_VERBOSE_MAKEFILE ON)


# Enable CTest for testing these code examples.
# include(CTest)

set(DSS_LIB "dss")
set(DSS_LIB_TEST "test_lib")

# Build shared libraries by default.
if(NOT BUILD_SHARED_LIBS)
    set(BUILD_SHARED_LIBS ON)
endif()

if(NOT NKV_SDK_DIR)
    set(NKV_SDK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../dss-sdk/host_out/")
endif()

set(NKV_INC_DIR "${NKV_SDK_DIR}/include")
set(NKV_LIB_DIR "${NKV_SDK_DIR}/lib")
message(STATUS "NKV_SDK_DIR: ${NKV_SDK_DIR}")

# Find the AWS SDK for C++ package.
find_package(AWSSDK REQUIRED COMPONENTS s3)
find_package(Python ${PYTHON_VERSION} REQUIRED COMPONENTS Interpreter)

execute_process(COMMAND python3-config --libs OUTPUT_VARIABLE PYTHON3_LINK_LIBRARIES_FLAG OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_STRIP_TRAILING_WHITESPACE)
separate_arguments(PYTHON3_LINK_LIBRARIES_FLAGS UNIX_COMMAND "${PYTHON3_LINK_LIBRARIES_FLAG}")

execute_process(COMMAND git describe --abbrev=4 --always --tags OUTPUT_VARIABLE GIT_COMMIT OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND python3-config --extension-suffix OUTPUT_VARIABLE EXT_NAME_SUFFIX)
execute_process(COMMAND python3 -m pybind11 --includes OUTPUT_VARIABLE EXT_INCLUDE_FLAG)
separate_arguments(EXT_INCLUDE_FLAGS UNIX_COMMAND "${EXT_INCLUDE_FLAG}")

option(WITH_RELEASE "release build" OFF)
if (WITH_RELEASE)
	SET(CMAKE_CXX_FLAGS "-O2 -Wall -Werror")
else ()
	SET(CMAKE_CXX_FLAGS "-O0 -g3 -Wall -Werror")
endif()
#message(STATUS "CXX FLAGS: ${CMAKE_CXX_FLAGS}")

set(SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/dss_client.cpp)

set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

#message(STATUS " FLAGS: ${CMAKE_CXX_FLAGS}")
add_executable(test_dss ${SOURCES} ${CMAKE_CURRENT_SOURCE_DIR}/test/test_dss_c_interface.c)
add_library(${DSS_LIB} SHARED ${SOURCES})

target_compile_definitions(${DSS_LIB} PUBLIC "DSS_DEBUG")
target_compile_definitions(test_dss PUBLIC "DSS_DEBUG")

target_include_directories(test_dss PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${NKV_INC_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>)
target_include_directories(${DSS_LIB} PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${NKV_INC_DIR}>
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>)

target_link_libraries(test_dss ${AWSSDK_LINK_LIBRARIES})
target_link_libraries(${DSS_LIB} ${AWSSDK_LINK_LIBRARIES})
target_link_libraries(test_dss "${PYTHON3_LINK_LIBRARIES_FLAGS} ${PYTHON_LIBRARIES}  -L${NKV_LIB_DIR} -lcrypto -lcurl -lrdd_cl")
target_link_libraries(${DSS_LIB} "${PYTHON3_LINK_LIBRARIES_FLAGS} ${PYTHON_LIBRARIES} -L${NKV_LIB_DIR} -lcrypto -lcurl -lrdd_cl")


target_compile_options(${DSS_LIB} PRIVATE ${EXT_INCLUDE_FLAGS})
target_compile_options(test_dss PRIVATE ${EXT_INCLUDE_FLAGS})
target_link_options(test_dss PUBLIC "-Wl,--disable-new-dtags" "-Wl,-rpath,/usr/lib/x86_64-linux-gnu/")

string(REGEX REPLACE "\\.[^.]*$" "" PYTHON_LIB_SUFFIX ${EXT_NAME_SUFFIX})
string(CONCAT DSS_PYTHON_LIB ${DSS_LIB} ${PYTHON_LIB_SUFFIX})

add_compile_definitions(GIT_VERSION=${GIT_COMMIT})

add_library(${DSS_PYTHON_LIB} SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/dss_ext.cpp)
target_include_directories(${DSS_PYTHON_LIB}  PUBLIC
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        	$<BUILD_INTERFACE:${NKV_INC_DIR}>
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>)

target_link_libraries(${DSS_PYTHON_LIB} "${PYTHON3_LINK_LIBRARIES_FLAGS} ${PYTHON_LIBRARIES} -L${NKV_LIB_DIR} -lcrypto -lcurl -lrdd_cl")

target_compile_options(${DSS_PYTHON_LIB} PRIVATE ${EXT_INCLUDE_FLAGS})
target_compile_options(${DSS_PYTHON_LIB} PRIVATE "-fvisibility=hidden")
target_link_options(${DSS_PYTHON_LIB} PUBLIC "-Wl,--disable-new-dtags" "-Wl,-rpath,/usr/lib/x86_64-linux-gnu/")

set_target_properties(${DSS_PYTHON_LIB} PROPERTIES PREFIX "")

# Installation
install(TARGETS ${DSS_LIB} DESTINATION ./lib64 )
install(TARGETS ${DSS_LIB} ${DSS_PYTHON_LIB} DESTINATION ${Python_SITEARCH})

