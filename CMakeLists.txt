PROJECT(cmake_addmc) # overwrites 'Makefile' (lower precedence than manually written 'GNUmakefile' wrt 'make')

CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12)

SET(CMAKE_CXX_STANDARD 14)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)

SET(CMAKE_CXX_FLAGS "-g -Ofast -static")

SET(lib_dir ${CMAKE_CURRENT_SOURCE_DIR}/addmc/libraries)

SET(cxxopts_target ${lib_dir}/cxxopts/include/cxxopts.hpp)

SET(cudd_dir ${lib_dir}/cudd)
SET(cudd_target ${cudd_dir}/config.h)

ADD_CUSTOM_COMMAND(
  OUTPUT ${cudd_target}
  COMMAND cd ${cudd_dir} && autoreconf && ./configure --silent --enable-obj && make -s
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

LINK_DIRECTORIES(${cudd_dir}/cudd/.libs)

INCLUDE_DIRECTORIES(${cudd_dir})
INCLUDE_DIRECTORIES(${cudd_dir}/cudd)
INCLUDE_DIRECTORIES(${cudd_dir}/epd)
INCLUDE_DIRECTORIES(${cudd_dir}/mtr)
INCLUDE_DIRECTORIES(${cudd_dir}/st)

FILE(GLOB cpp_files "addmc/src/implementation/*.cpp")

ADD_EXECUTABLE(addmc ${cpp_files} ${cxxopts_target} ${cudd_target})

TARGET_LINK_LIBRARIES(addmc cudd)
