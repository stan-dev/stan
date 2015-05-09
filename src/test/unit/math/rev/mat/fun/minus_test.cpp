#include <stan/math/prim/mat/meta/get.hpp>
#include <stan/math/prim/arr/meta/get.hpp>
#include <stan/math/prim/mat/meta/length.hpp>
#include <stan/math/prim/mat/meta/is_vector.hpp>
#include <stan/math/prim/mat/meta/is_vector_like.hpp>
#include <stan/math/prim/mat/fun/minus.hpp>
#include <gtest/gtest.h>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <stan/math/rev/mat/fun/typedefs.hpp>
#include <stan/math/rev/core.hpp>

TEST(AgradRevMatrix, minus_scalar) {
  using stan::math::minus;
  double x = 10;
  AVAR v = 11;
  
  EXPECT_FLOAT_EQ(-10, minus(x));
  EXPECT_FLOAT_EQ(-11, minus(v).val());
}
TEST(AgradRevMatrix, minus_vector) {
  using stan::math::vector_d;
  using stan::math::vector_v;
  using stan::math::minus;

  vector_d d(3);
  vector_v v(3);

  d << -100, 0, 1;
  v << -100, 0, 1;
  
  vector_d output_d;
  output_d = minus(d);
  EXPECT_FLOAT_EQ(100, output_d[0]);
  EXPECT_FLOAT_EQ(0, output_d[1]);
  EXPECT_FLOAT_EQ(-1, output_d[2]);

  vector_v output;
  output = minus(v);
  EXPECT_FLOAT_EQ(100, output[0].val());
  EXPECT_FLOAT_EQ(0, output[1].val());
  EXPECT_FLOAT_EQ(-1, output[2].val());
}
TEST(AgradRevMatrix, minus_rowvector) {
  using stan::math::row_vector_d;
  using stan::math::row_vector_v;
  using stan::math::minus;

  row_vector_d d(3);
  row_vector_v v(3);

  d << -100, 0, 1;
  v << -100, 0, 1;
  
  row_vector_d output_d;
  output_d = minus(d);
  EXPECT_FLOAT_EQ(100, output_d[0]);
  EXPECT_FLOAT_EQ(0, output_d[1]);
  EXPECT_FLOAT_EQ(-1, output_d[2]);

  row_vector_v output;
  output = minus(v);
  EXPECT_FLOAT_EQ(100, output[0].val());
  EXPECT_FLOAT_EQ(0, output[1].val());
  EXPECT_FLOAT_EQ(-1, output[2].val());
}
TEST(AgradRevMatrix, minus_matrix) {
  using stan::math::matrix_d;
  using stan::math::matrix_v;
  using stan::math::minus;

  matrix_d d(2, 3);
  matrix_v v(2, 3);

  d << -100, 0, 1, 20, -40, 2;
  v << -100, 0, 1, 20, -40, 2;

  matrix_d output_d = minus(d);
  EXPECT_FLOAT_EQ(100, output_d(0,0));
  EXPECT_FLOAT_EQ(  0, output_d(0,1));
  EXPECT_FLOAT_EQ( -1, output_d(0,2));
  EXPECT_FLOAT_EQ(-20, output_d(1,0));
  EXPECT_FLOAT_EQ( 40, output_d(1,1));
  EXPECT_FLOAT_EQ( -2, output_d(1,2));

  matrix_v output = minus(v);
  EXPECT_FLOAT_EQ(100, output(0,0).val());
  EXPECT_FLOAT_EQ(  0, output(0,1).val());
  EXPECT_FLOAT_EQ( -1, output(0,2).val());
  EXPECT_FLOAT_EQ(-20, output(1,0).val());
  EXPECT_FLOAT_EQ( 40, output(1,1).val());
  EXPECT_FLOAT_EQ( -2, output(1,2).val());
}
