#include <stan/mcmc/mcmc_output.hpp>
#include <gtest/gtest.h>



TEST(stanMcmc, effectivesamplesize)  {
  // R code:
  // R> set.seed(0)
  // R> y1 <- rnorm(10)
  // R> y2 <- rnorm(10)
  // R> y <- cbind(y1, y2)
  std::vector< std::vector<double> > y;
  std::vector<double> y1;
  y1.push_back(1.262954285);
  y1.push_back(-0.326233361);
  y1.push_back(1.329799263);
  y1.push_back(1.272429321);
  y1.push_back(0.414641434);
  y1.push_back(-1.539950042);
  y1.push_back(0.928567035);
  y1.push_back(-0.294720447);
  y1.push_back(-0.005767173);
  y1.push_back(2.404653389);
  std::vector<double> y2;
  y2.push_back(0.7635935);
  y2.push_back(-0.7990092);
  y2.push_back(-1.1476570);
  y2.push_back(-0.2894616);
  y2.push_back(-0.2992151);
  y2.push_back(-0.4115108);
  y2.push_back(0.2522234);
  y2.push_back(-0.8919211);
  y2.push_back(0.4356833);
  y2.push_back(-1.2375384);
    
  y.push_back(y1);
  y.push_back(y2);

  EXPECT_FLOAT_EQ(10, stan::mcmc::ess(y));
}
