#include <gtest/gtest.h>
#include <stan/callbacks/noop_interrupt.hpp>

TEST(StanCallbacks, op) {
  stan::callbacks::noop_interrupt interrupt;

  EXPECT_NO_THROW(interrupt());
}
