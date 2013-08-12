#include <gtest/gtest.h>
#include <stan/diff/fvar.hpp>
#include <stan/math/functions/Phi.hpp>
#include <stan/prob/distributions/univariate/continuous/normal.hpp>

TEST(AgradFvar,Phi) {
  using stan::diff::fvar;
  using stan::math::Phi;
  fvar<double> x = 1.0;
  
  fvar<double> Phi_x = Phi(x);

  EXPECT_FLOAT_EQ(Phi(1.0), Phi_x.val_);
  EXPECT_FLOAT_EQ(exp(stan::prob::normal_log<false>(1.0,0.0,1.0)),
                  Phi_x.d_);
}
