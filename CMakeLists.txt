cmake_minimum_required(VERSION 3.26)
project(CorinthoAI)

set(CMAKE_CXX_STANDARD 17)

set(GTEST_PATH "googletest/googletest")

set(GSL_PATH "gsl")

set(CPP_PATH "corintho_ai/cpp")

set(TEST_PATH "tests/cpp")

set(CMAKE_CXX_FLAGS "--coverage -fopenmp -g -pg -O3 -fopt-info-vec-optimized")

include_directories(${GTEST_PATH}/include ${GSL_PATH}/include ${CPP_PATH}/include)
link_directories(googletest/lib)

add_executable(
    CorinthoAI ${TEST_PATH}/main.cpp
    ${TEST_PATH}/move_test.cpp ${TEST_PATH}/game_test.cpp ${TEST_PATH}/node_test.cpp
    ${TEST_PATH}/trainmc_test.cpp ${TEST_PATH}/selfplayer_test.cpp ${TEST_PATH}/trainer_test.cpp ${TEST_PATH}/match_test.cpp
    ${CPP_PATH}/src/util.cpp ${CPP_PATH}/src/move.cpp ${CPP_PATH}/src/game.cpp ${CPP_PATH}/src/node.cpp
    ${CPP_PATH}/src/trainmc.cpp ${CPP_PATH}/src/selfplayer.cpp ${CPP_PATH}/src/trainer.cpp ${CPP_PATH}/src/match.cpp
)
target_link_libraries(CorinthoAI gtest gtest_main pthread)