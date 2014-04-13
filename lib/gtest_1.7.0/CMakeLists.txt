set(GTEST_SRCS 
    src/gtest-all.cc
    src/gtest_main.cc)

include_directories(include .)

add_library(gtest STATIC ${GTEST_SRCS})

