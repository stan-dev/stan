#include <stan/math/memory/stack_alloc.hpp>
#include <gtest/gtest.h>

TEST(stack_alloc, speed_of_allocator) {
  stan::math::stack_alloc allocator;
  for (size_t m = 0; m < 10000; m++) {
    for (size_t n = 1; n <= 100000; ++n) {
      allocator.alloc(n);
    }
    allocator.recover_all();
  }
}
