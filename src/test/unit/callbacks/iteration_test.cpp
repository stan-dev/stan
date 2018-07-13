#include <gtest/gtest.h>
#include <stan/callbacks/iteration.hpp>

TEST(StanCallbacks, op) {
  stan::callbacks::iteration iteration;

  EXPECT_NO_THROW(iteration(1));
}
