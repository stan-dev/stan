#include <gtest/gtest.h>
#include <stan/math/matrix/quad_form.hpp>
#include <stan/math/matrix/typedefs.hpp>

TEST(MathMatrix, quad_form_mat) {
  using stan::math::quad_form;
  using stan::math::matrix_d;
  
  matrix_d ad(4,4);
  matrix_d bd(4,2);
  
  
  bd << 100, 10,
          0,  1,
         -3, -3,
          5,  2;
  ad << 2.0,  3.0, 4.0,   5.0, 
  6.0, 10.0, 2.0,   2.0,
  7.0,  2.0, 7.0,   1.0,
  8.0,  2.0, 1.0, 112.0;
  
  // double-double
  matrix_d resd = quad_form(ad,bd);
  EXPECT_FLOAT_EQ(26033, resd(0,0));
  EXPECT_FLOAT_EQ(3456, resd(0,1));
  EXPECT_FLOAT_EQ(3396, resd(1,0));
  EXPECT_FLOAT_EQ(725, resd(1,1));
}

TEST(MathMatrix, quad_form_sym_mat) {
  using stan::math::quad_form_sym;
  using stan::math::matrix_d;
  
  matrix_d ad(4,4);
  matrix_d bd(4,2);
  
  bd << 100, 10,
  0,  1,
  -3, -3,
  5,  2;
  ad << 2.0,  3.0, 4.0,   5.0, 
  3.0, 10.0, 2.0,   2.0,
  4.0,  2.0, 7.0,   1.0,
  5.0,  2.0, 1.0, 112.0;
  
  // double-double
  matrix_d resd = quad_form_sym(ad,bd);
  EXPECT_FLOAT_EQ(25433, resd(0,0));
  EXPECT_FLOAT_EQ(3396, resd(0,1));
  EXPECT_FLOAT_EQ(3396, resd(1,0));
  EXPECT_FLOAT_EQ(725, resd(1,1));
}

TEST(MathMatrix, quad_form_vec) {
  using stan::math::quad_form;
  using stan::math::matrix_d;
  using stan::math::vector_d;

  matrix_d ad(4,4);
  vector_d bd(4);
  double res;

  
  bd << 100, 0, -3, 5;
  ad << 2.0,  3.0, 4.0,   5.0, 
  6.0, 10.0, 2.0,   2.0,
  7.0,  2.0, 7.0,   1.0,
  8.0,  2.0, 1.0, 112.0;
  
  // double-double
  res = quad_form(ad,bd);
  EXPECT_FLOAT_EQ(26033, res);
}

TEST(MathMatrix, quad_form_sym_vec) {
  using stan::math::quad_form_sym;
  using stan::math::matrix_d;
  using stan::math::vector_d;
  
  matrix_d ad(4,4);
  vector_d bd(4);
  double res;
  
  bd << 100, 0, -3, 5;
  ad << 2.0,  3.0, 4.0,   5.0, 
  3.0, 10.0, 2.0,   2.0,
  4.0,  2.0, 7.0,   1.0,
  5.0,  2.0, 1.0, 112.0;
  
  // double-double
  res = quad_form_sym(ad,bd);
  EXPECT_FLOAT_EQ(25433, res);
}
