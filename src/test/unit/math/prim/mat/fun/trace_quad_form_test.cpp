#include <stan/math/prim/mat/fun/trace_quad_form.hpp>
#include <gtest/gtest.h>
#include <stan/math/prim/mat/fun/typedefs.hpp>

TEST(MathMatrix, trace_quad_form_mat) {
  using stan::math::trace_quad_form;
  using stan::math::matrix_d;
  
  matrix_d ad(4,4);
  matrix_d bd(4,2);
  double res;
  bd << 100, 10,
          0,  1,
         -3, -3,
          5,  2;
  ad << 2.0,  3.0, 4.0,   5.0, 
  6.0, 10.0, 2.0,   2.0,
  7.0,  2.0, 7.0,   1.0,
  8.0,  2.0, 1.0, 112.0;
  
  // double-double
  res = trace_quad_form(ad,bd);
  EXPECT_FLOAT_EQ(26758, res);
}
