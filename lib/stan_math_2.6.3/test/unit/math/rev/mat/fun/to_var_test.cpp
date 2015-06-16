#include <stan/math/prim/mat/meta/get.hpp>
#include <stan/math/prim/arr/meta/get.hpp>
#include <stan/math/prim/mat/meta/length.hpp>
#include <stan/math/prim/mat/meta/is_vector.hpp>
#include <stan/math/prim/mat/meta/is_vector_like.hpp>
#include <stan/math/rev/mat/fun/to_var.hpp>
#include <gtest/gtest.h>
#include <test/unit/math/rev/mat/fun/util.hpp>

TEST(AgradRevMatrix,to_var_scalar) {
  double d = 5.0;
  AVAR v = 5.0;
  stan::math::var var_x = stan::math::to_var(d);
  EXPECT_FLOAT_EQ(5.0, var_x.val());

  var_x = stan::math::to_var(v);
  EXPECT_FLOAT_EQ(5.0, var_x.val());
}
TEST(AgradRevMatrix,to_var_matrix) {
  using stan::math::matrix_d;
  using stan::math::matrix_v;
  matrix_d m_d(2,3);
  m_d << 0, 1, 2, 3, 4, 5;
  matrix_v m_v = stan::math::to_var(m_d);
  
  EXPECT_EQ(2, m_v.rows());
  EXPECT_EQ(3, m_v.cols());
  for (int ii = 0; ii < 2; ii++) 
    for (int jj = 0; jj < 3; jj++)
      EXPECT_FLOAT_EQ(ii*3 + jj, m_v(ii, jj).val());
}
TEST(AgradRevMatrix,to_var_vector) {
  using stan::math::vector_d;
  using stan::math::vector_v;

  vector_d d(5);
  vector_v v(5);
  
  d << 1, 2, 3, 4, 5;
  v << 1, 2, 3, 4, 5;
  
  vector_v out = stan::math::to_var(d);
  EXPECT_FLOAT_EQ(1, out(0).val());
  EXPECT_FLOAT_EQ(2, out(1).val());
  EXPECT_FLOAT_EQ(3, out(2).val());
  EXPECT_FLOAT_EQ(4, out(3).val());
  EXPECT_FLOAT_EQ(5, out(4).val());

  out = stan::math::to_var(v);
  EXPECT_FLOAT_EQ(1, out(0).val());
  EXPECT_FLOAT_EQ(2, out(1).val());
  EXPECT_FLOAT_EQ(3, out(2).val());
  EXPECT_FLOAT_EQ(4, out(3).val());
  EXPECT_FLOAT_EQ(5, out(4).val());
}
TEST(AgradRevMatrix,to_var_rowvector) {
  using stan::math::row_vector_d;
  using stan::math::row_vector_v;

  row_vector_d d(5);
  row_vector_v v(5);
  
  d << 1, 2, 3, 4, 5;
  v << 1, 2, 3, 4, 5;
  
  row_vector_v output = stan::math::to_var(d);
  EXPECT_FLOAT_EQ(1, output(0).val());
  EXPECT_FLOAT_EQ(2, output(1).val());
  EXPECT_FLOAT_EQ(3, output(2).val());
  EXPECT_FLOAT_EQ(4, output(3).val());
  EXPECT_FLOAT_EQ(5, output(4).val());

  output.resize(0);
  output = stan::math::to_var(v);
  EXPECT_FLOAT_EQ(1, output(0).val());
  EXPECT_FLOAT_EQ(2, output(1).val());
  EXPECT_FLOAT_EQ(3, output(2).val());
  EXPECT_FLOAT_EQ(4, output(3).val());
  EXPECT_FLOAT_EQ(5, output(4).val());
}

