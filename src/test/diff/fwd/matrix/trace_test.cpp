#include <stan/math/matrix/trace.hpp>
#include <gtest/gtest.h>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/agrad/fwd/matrix/typedefs.hpp>
#include <stan/agrad/fvar.hpp>

TEST(AgradFwdMatrix,mv_trace) {
  using stan::math::trace;
  using stan::agrad::matrix_fv;
  using stan::agrad::fvar;

  matrix_fv a(2,2);
  a << -1.0, 2.0, 
    5.0, 10.0;
   a(0,0).d_ = 1.0;
   a(0,1).d_ = 1.0;
   a(1,0).d_ = 1.0;
   a(1,1).d_ = 1.0;
  
  fvar<double> s = trace(a);
  EXPECT_FLOAT_EQ(9.0,s.val_);
  EXPECT_FLOAT_EQ(2.0,s.d_);
}  
