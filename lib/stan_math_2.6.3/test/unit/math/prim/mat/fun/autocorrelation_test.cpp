#include <fstream>
#include <vector>
#include <stdexcept>

#include <gtest/gtest.h>

#include <stan/math/prim/mat/fun/autocorrelation.hpp>

TEST(ProbAutocorrelation,test1) {
  // ar1.csv generated in R with
  //   > x[1] <- rnorm(1,0,1)
  //   > for (n in 2:1000) x[n] <- rnorm(1,0.8 * x[n-1],1)
  std::fstream f("test/unit/math/prim/mat/fun/ar1.csv");
  std::vector<double> y;
  for (size_t i = 0; i < 1000; ++i) {
     double temp;
     f >> temp;
     y.push_back(temp);
   }

   // 10K 1K-length AC in 2.9s with g++ -O3 on Bob's Macbook Air
   std::vector<double> ac;

   size_t ITS = 1;  // only need one for test
   for (size_t n = 0; n < ITS; ++n) {
     stan::math::autocorrelation(y,ac);
   }

   EXPECT_EQ(1000U, ac.size());

   EXPECT_NEAR(1.00, ac[0],0.001);
   EXPECT_NEAR(0.80, ac[1], 0.01);
   EXPECT_NEAR(0.64, ac[2], 0.01);
   EXPECT_NEAR(0.51, ac[3], 0.01);
   EXPECT_NEAR(0.41, ac[4], 0.01);
   EXPECT_NEAR(0.33, ac[5], 0.01);
}

TEST(ProbAutocorrelation,fft_next_good_size) {
  EXPECT_EQ(2U, stan::math::fft_next_good_size(0));
  EXPECT_EQ(2U, stan::math::fft_next_good_size(1));
  EXPECT_EQ(2U, stan::math::fft_next_good_size(2));
  EXPECT_EQ(3U, stan::math::fft_next_good_size(3));
  
  EXPECT_EQ(4U, stan::math::fft_next_good_size(4));
  EXPECT_EQ(128U, stan::math::fft_next_good_size(128));
  EXPECT_EQ(135U, stan::math::fft_next_good_size(129));
}
