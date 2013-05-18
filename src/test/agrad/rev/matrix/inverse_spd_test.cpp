#include <stan/math/matrix/inverse_spd.hpp>
#include <stan/math/matrix/multiply.hpp>
#include <gtest/gtest.h>
#include <test/agrad/util.hpp>
#include <stan/agrad/agrad.hpp>
#include <stan/agrad/matrix.hpp>
#include <stan/agrad/rev/print_stack.hpp>

TEST(AgradRevMatrix,inverse_spd_val) {
  using stan::math::inverse_spd;
  using stan::agrad::matrix_v;

  matrix_v a(2,2);
  a << 2.0, 3.0, 
    3.0, 7.0;

  matrix_v a_inv = inverse_spd(a);

  matrix_v I = multiply(a,a_inv);

  EXPECT_NEAR(1.0,I(0,0).val(),1.0E-12);
  EXPECT_NEAR(0.0,I(0,1).val(),1.0E-12);
  EXPECT_NEAR(0.0,I(1,0).val(),1.0E-12);
  EXPECT_NEAR(1.0,I(1,1).val(),1.0e-12);

  EXPECT_THROW(inverse_spd(matrix_v(2,3)), std::domain_error);

  a << 2.0, 3.0, 
  1.0, 7.0;
  EXPECT_THROW(inverse_spd(a), std::domain_error);
  a << 1.0, -1.0, 
  -1.0, -1.0;
  EXPECT_THROW(inverse_spd(a), std::domain_error);
}

TEST(AgradRevMatrix,inverse_spd_inverse_spd_sum) {
  using stan::math::sum;
  using stan::math::inverse_spd;
  using stan::agrad::matrix_v;
  
  matrix_v a(4,4);
  a << 1.0, 0.0, 0.0, 0.0, 
  0.0, 1.0, 0.0, 0.0,
  0.0, 0.0, 1.0, 0.0,
  0.0, 0.0, 0.0, 1.0;
  
  AVEC x;
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j)
      x.push_back(a(i,j));
  
  AVAR a_inv_inv_sum = sum(inverse_spd(inverse_spd(a)));
  
  VEC g;
  a_inv_inv_sum.grad(x,g);

  size_t k = 0;
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      if (i == j)
        EXPECT_FLOAT_EQ(1.0,g[k]);
      else if (i > j)
        EXPECT_FLOAT_EQ(2.0,g[k]);
      else
        EXPECT_FLOAT_EQ(0.0,g[k]);
      k++;
    }
  }
}
