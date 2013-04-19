#include <stan/math/matrix/exp.hpp>
#include <gtest/gtest.h>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/agrad/fwd/matrix/typedefs.hpp>
#include <stan/agrad/fwd/exp.hpp>

TEST(AgradFwdMatrix, exp_matrix) {
  using stan::math::exp;
  using stan::math::matrix_d;
  using stan::agrad::matrix_fv;

  matrix_d expected_output(2,2);
  matrix_fv mv(2,2), output;
  int i,j;

  mv << 1, 2, 3, 4;
   mv(0).d_ = 2.0;
   mv(1).d_ = 2.0;
   mv(2).d_ = 2.0;
   mv(3).d_ = 2.0;
  expected_output << std::exp(1), std::exp(2), std::exp(3), std::exp(4);
  output = exp(mv);

  for (i = 0; i < 2; i++)
    for (j = 0; j < 2; j++)
      EXPECT_FLOAT_EQ(expected_output(i,j), output(i,j).val_);
  EXPECT_FLOAT_EQ(2.0 * std::exp(1), output(0,0).d_);
  EXPECT_FLOAT_EQ(2.0 * std::exp(2), output(0,1).d_);
  EXPECT_FLOAT_EQ(2.0 * std::exp(3), output(1,0).d_);
  EXPECT_FLOAT_EQ(2.0 * std::exp(4), output(1,1).d_);
}
TEST(AgradFwdMatrix, exp_vector) {
  using stan::math::exp;
  using stan::math::vector_d;
  using stan::agrad::vector_fv;

  vector_d expected_output(4);
  vector_fv mv(4), output;

  mv << 1, 2, 3, 4;
   mv(0).d_ = 2.0;
   mv(1).d_ = 2.0;
   mv(2).d_ = 2.0;
   mv(3).d_ = 2.0;
  expected_output << std::exp(1), std::exp(2), std::exp(3), std::exp(4);
  output = exp(mv);

  for (int i = 0; i < 4; i++)
    EXPECT_FLOAT_EQ(expected_output(i), output(i).val_);
  EXPECT_FLOAT_EQ(2.0 * std::exp(1), output(0).d_);
  EXPECT_FLOAT_EQ(2.0 * std::exp(2), output(1).d_);
  EXPECT_FLOAT_EQ(2.0 * std::exp(3), output(2).d_);
  EXPECT_FLOAT_EQ(2.0 * std::exp(4), output(3).d_);
}
TEST(AgradFwdMatrix, exp_rowvector) {
  using stan::math::exp;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_fv;

  row_vector_d expected_output(4);
  row_vector_fv mv(4), output;

  mv << 1, 2, 3, 4;
   mv(0).d_ = 2.0;
   mv(1).d_ = 2.0;
   mv(2).d_ = 2.0;
   mv(3).d_ = 2.0;
  expected_output << std::exp(1), std::exp(2), std::exp(3), std::exp(4);
  output = exp(mv);

  for (int i = 0; i < 4; i++)
    EXPECT_FLOAT_EQ(expected_output(i), output(i).val_);
  EXPECT_FLOAT_EQ(2.0 * std::exp(1), output(0).d_);
  EXPECT_FLOAT_EQ(2.0 * std::exp(2), output(1).d_);
  EXPECT_FLOAT_EQ(2.0 * std::exp(3), output(2).d_);
  EXPECT_FLOAT_EQ(2.0 * std::exp(4), output(3).d_);
}
