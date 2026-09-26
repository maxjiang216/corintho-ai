cmake_minimum_required(VERSION 3.26)

# clang++ 20 by default when it is installed (worklog entry 28); must be set
# before project(). -DCMAKE_CXX_COMPILER=... or CXX in the environment still
# override it, and without clang++-20 (CI) CMake picks its usual compiler.
if(NOT DEFINED CMAKE_CXX_COMPILER AND NOT DEFINED ENV{CXX})
  find_program(CLANGXX_20 clang++-20)
  if(CLANGXX_20)
    set(CMAKE_CXX_COMPILER ${CLANGXX_20})
  endif()
endif()

project(CorinthoAI)

set(CMAKE_CXX_STANDARD 17)

set(GTEST_PATH "googletest/googletest")

set(GSL_PATH "gsl")

set(CPP_PATH "corintho_ai/cpp")

set(TEST_PATH "tests/cpp")

set(CMAKE_CXX_FLAGS "--coverage -fopenmp -g -pg -O3")

include_directories(${GTEST_PATH}/include ${GSL_PATH}/include ${CPP_PATH}/include)
link_directories(googletest/lib)

add_executable(
    CorinthoAI ${TEST_PATH}/main.cpp
    ${TEST_PATH}/move_test.cpp ${TEST_PATH}/game_test.cpp ${TEST_PATH}/node_test.cpp
    ${TEST_PATH}/trainmc_test.cpp ${TEST_PATH}/selfplayer_test.cpp ${TEST_PATH}/trainer_test.cpp
    ${TEST_PATH}/match_test.cpp ${TEST_PATH}/tourney_test.cpp ${TEST_PATH}/solver_test.cpp
    ${CPP_PATH}/src/util.cpp ${CPP_PATH}/src/move.cpp ${CPP_PATH}/src/game.cpp ${CPP_PATH}/src/node.cpp
    ${CPP_PATH}/src/trainmc.cpp ${CPP_PATH}/src/selfplayer.cpp ${CPP_PATH}/src/trainer.cpp
    ${CPP_PATH}/src/match.cpp ${CPP_PATH}/src/tourney.cpp ${CPP_PATH}/src/solver.cpp ${CPP_PATH}/src/solver_pool.cpp
)
target_link_libraries(CorinthoAI gtest gtest_main pthread)