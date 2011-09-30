#include <vector>
#include <gtest/gtest.h>
#include "stan/prob/rhat.hpp"

TEST(prob_rhat,construct) {
  stan::prob::rhat rh(3,1);
  EXPECT_FLOAT_EQ(0,0);
  
  std::vector<double> x(1);
  
  EXPECT_EQ(3, rh.num_chains());

  x[0] = 1.0;
  rh.add(0,x);
  x[0] = 1.2;
  rh.add(0,x);
  x[0] = 1.0;
  rh.add(0,x);
  x[0] = 0.9;
  rh.add(0,x);
  x[0] = 0.6;
  rh.add(0,x);

  x[0] = 0.5;
  rh.add(1,x);
  x[0] = 1.0;
  rh.add(1,x);
  x[0] = 0.9;
  rh.add(1,x);
  x[0] = 1.3;
  rh.add(1,x);
  x[0] = 1.9;
  rh.add(1,x);

  x[0] = 1.2;
  rh.add(2,x);
  x[0] = 1.2;
  rh.add(2,x);
  x[0] = 1.5;
  rh.add(2,x);
  x[0] = 1.0;
  rh.add(2,x);
  x[0] = 1.6;
  rh.add(2,x);

  rh.compute(x);
  
  
  EXPECT_FLOAT_EQ(1.027516,x[0]);
  

}
