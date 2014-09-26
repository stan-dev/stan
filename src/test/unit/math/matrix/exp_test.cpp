#include <stan/math/matrix/exp.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <gtest/gtest.h>

TEST(MathMatrix, exp) {
  using stan::math::exp;
  stan::math::matrix_d expected_output(2,2);
  stan::math::matrix_d mv(2,2), output;
  int i,j;

  mv << 1, 2, 3, 4;
  expected_output << std::exp(1), std::exp(2), std::exp(3), std::exp(4);
  output = exp(mv);

  for (i = 0; i < 2; i++)
    for (j = 0; j < 2; j++)
      EXPECT_FLOAT_EQ(expected_output(i,j), output(i,j));
}

TEST(MathMatrix, exp_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();

  stan::math::matrix_d m1(2,2);
  m1 << 1, nan,
        3, 6;
        
  stan::math::matrix_d mr;

  using stan::math::exp;
  using boost::math::isnan;
  
  mr = exp(m1);
  
  EXPECT_DOUBLE_EQ(std::exp(1), mr(0, 0));
  EXPECT_PRED1(isnan<double>, mr(0, 1));
  EXPECT_DOUBLE_EQ(std::exp(3), mr(1, 0));
  EXPECT_DOUBLE_EQ(std::exp(6), mr(1, 1));
}
