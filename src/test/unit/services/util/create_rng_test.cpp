#include <stan/services/util/create_rng.hpp>
#include <gtest/gtest.h>

TEST(rng, initialize_with_seed) {
  stan::rng_t rng1 = stan::services::util::create_rng(0, 1);
  stan::rng_t rng2 = stan::services::util::create_rng(0, 1);
  EXPECT_EQ(rng1, rng2);

  rng2();  // generate a random number
  EXPECT_NE(rng1, rng2);
}

TEST(rng, initialize_with_id) {
  stan::rng_t rng1 = stan::services::util::create_rng(0, 1);
  for (unsigned int n = 2; n < 20; n++) {
    stan::rng_t rng2 = stan::services::util::create_rng(0, n);
    EXPECT_NE(rng1, rng2);
  }
}

// warning---this will reuse draws from transformed data
// if we initialize with zero
TEST(rng, initialize_with_zero) {
  stan::rng_t rng1 = stan::services::util::create_rng(0, 0);
  stan::rng_t rng2 = stan::services::util::create_rng(0, 0);
  EXPECT_EQ(rng1, rng2);

  rng2();
  EXPECT_NE(rng1, rng2);
}
