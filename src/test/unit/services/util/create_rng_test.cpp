#include <stan/services/util/create_rng.hpp>
#include <gtest/gtest.h>

TEST(rng, initialize_with_seed) {
  boost::ecuyer1988 rng1 = stan::services::util::create_rng(0, 1);
  boost::ecuyer1988 rng2 = stan::services::util::create_rng(0, 1);
  EXPECT_EQ(rng1, rng2);

  rng2();  // generate a random number
  EXPECT_NE(rng1, rng2);
}

TEST(rng, initialize_with_id) {
  boost::ecuyer1988 rng1 = stan::services::util::create_rng(0, 1);
  for (unsigned int n = 2; n < 20; n++) {
    boost::ecuyer1988 rng2 = stan::services::util::create_rng(0, n);
    EXPECT_NE(rng1, rng2);
  }
}

// warning---this will reuse draws from transformed data
// if we initialize with zero
TEST(rng, initialize_with_zero) {
  boost::ecuyer1988 rng1 = stan::services::util::create_rng(0, 0);
  boost::ecuyer1988 rng2 = stan::services::util::create_rng(0, 0);
  EXPECT_EQ(rng1, rng2);

  rng2();
  EXPECT_NE(rng1, rng2);
}
