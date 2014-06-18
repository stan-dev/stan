#include <gtest/gtest.h>
#include <stan/prob/internal_math/fwd/inc_beta.hpp>
#include <stan/agrad/fwd/fvar.hpp>

TEST(ProbInternalMath, inc_beta_fd) {
  using stan::agrad::fvar;
  fvar<double> a = 1.0;
  fvar<double> b = 1.0;
  fvar<double> g = 0.4;
  a.d_ = 1.0;
  b.d_ = 1.0;
  g.d_ = 1.0;
  
  EXPECT_FLOAT_EQ(0.4, stan::agrad::inc_beta(a, b, g).val_);
  EXPECT_NEAR(0.1399790720133741432386770490663565874456006538094571344439, 
              stan::agrad::inc_beta(a, b, g).d_,1e-6);
}
TEST(ProbInternalMath, inc_beta_ffd) {
  using stan::agrad::fvar;
  fvar<fvar<double> > a = 1.0;
  fvar<fvar<double> > b = 1.0;
  fvar<fvar<double> > g = 0.4;
  a.d_ = 1.0;
  b.d_ = 1.0;
  g.d_ = 1.0;
  
  EXPECT_FLOAT_EQ(0.4, stan::agrad::inc_beta(a, b, g).val_.val_);
  EXPECT_NEAR(0.1399790720133741432386770490663565874456006538094571344439, 
              stan::agrad::inc_beta(a, b, g).d_.val_,1e-6);
}
