#include <stan/math/prim/scal/prob/normal_log.hpp>
#include <gtest/gtest.h>
#include <stan/math/rev/scal/fun/value_of.hpp>
#include <stan/math/rev/scal/fun/value_of_rec.hpp>
#include <stan/math/rev/core.hpp>

TEST(ProbDistributionsNormal, intVsDouble) {
  using stan::math::var;
  for (double thetaval = -5.0; thetaval < 6.0; thetaval += 0.5) {
    var theta(thetaval);
    var lp1(0.0);
    lp1 += stan::math::normal_log<true>(0, theta, 1);
    double lp1val = lp1.val();
    stan::math::grad(lp1.vi_); 
    double lp1adj = lp1.adj();

    var theta2(thetaval);
    var lp2(0.0);
    lp2 += stan::math::normal_log<true>(theta2, 0, 1);
    double lp2val = lp2.val();
    stan::math::grad(lp2.vi_);
    double lp2adj = lp2.adj();
    EXPECT_FLOAT_EQ(lp1val,lp2val);
    EXPECT_FLOAT_EQ(lp1adj,lp2adj);
  }
}

