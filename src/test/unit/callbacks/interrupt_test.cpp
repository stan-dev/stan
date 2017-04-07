#include <gtest/gtest.h>
#include <stan/callbacks/interrupt.hpp>

TEST(StanCallbacks, op) {
  stan::callbacks::interrupt interrupt;

  EXPECT_NO_THROW(interrupt());
}
