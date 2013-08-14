#include <stan/math/matrix/trace.hpp>
#include <gtest/gtest.h>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/diff/fwd/matrix/typedefs.hpp>
#include <stan/diff/fwd.hpp>

TEST(DiffFwdMatrix,mv_trace) {
  using stan::math::trace;
  using stan::diff::matrix_fv;
  using stan::diff::fvar;

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
