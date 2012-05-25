#include <fstream>
#include <vector>
#include <stdexcept>

#include <gtest/gtest.h>

#include <stan/prob/autocorrelation.hpp>

TEST(ProbAutocorrelation,test1) {
  // ar1.csv generated in R with
  //   > x[1] <- rnorm(1,0,1)
  //   > for (n in 2:1000) x[n] <- rnorm(1,0.8 * x[n-1],1)
  std::fstream f("src/test/prob/ar1.csv");
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
     stan::prob::auto_correlation(y,ac);
   }

   EXPECT_EQ(1000, ac.size());

   EXPECT_NEAR(1.0,ac[0],0.001);
   EXPECT_NEAR(0.81, ac[1], 0.01);
   EXPECT_NEAR(0.65, ac[2], 0.01);
   EXPECT_NEAR(0.52, ac[3], 0.01);
   EXPECT_NEAR(0.42, ac[4], 0.01);
   EXPECT_NEAR(0.34, ac[5], 0.01);

   // std::cout << "ac.size()=" << ac.size() << std::endl;
   // for (size_t n = 0; n < ac.size(); ++n)
   //     std::cout << "ac[" << n << "]=" << ac[n] << std::endl;
}

