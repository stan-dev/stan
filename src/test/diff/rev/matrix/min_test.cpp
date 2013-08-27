#include <stan/math/matrix/min.hpp>
#include <gtest/gtest.h>
#include <test/diff/util.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/diff/rev/matrix/typedefs.hpp>
#include <stan/diff.hpp>

TEST(DiffRevMatrix, min_vector) {
  using stan::math::min;
  using stan::math::vector_d;
  using stan::diff::vector_v;

  vector_d d1(3);
  vector_v v1(3);
  
  d1 << 100, 0, -3;
  v1 << 100, 0, -3;
  
  AVAR output;
  output = min(d1);
  EXPECT_FLOAT_EQ(-3, output.val());
                   
  output = min(v1);
  EXPECT_FLOAT_EQ(-3, output.val());
}
TEST(DiffRevMatrix, min_vector_exception) {
  using stan::math::min;
  using stan::math::vector_d;
  using stan::diff::vector_v;

  vector_d d;
  vector_v v;
  d.resize(0);
  v.resize(0);
  EXPECT_EQ(std::numeric_limits<double>::infinity(), min(v).val());
}
TEST(DiffRevMatrix, min_rowvector) {
  using stan::math::min;
  using stan::math::row_vector_d;
  using stan::diff::row_vector_v;

  row_vector_d d1(3);
  row_vector_v v1(3);
  
  d1 << 100, 0, -3;
  v1 << 100, 0, -3;
  
  AVAR output;
  output = min(d1);
  EXPECT_FLOAT_EQ(-3, output.val());
                   
  output = min(v1);
  EXPECT_FLOAT_EQ(-3, output.val());
}
TEST(DiffRevMatrix, min_rowvector_exception) {
  using stan::math::min;
  using stan::diff::row_vector_v;

  row_vector_v v;
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), min(v).val());
}
TEST(DiffRevMatrix, min_matrix) {
  using stan::math::min;
  using stan::math::matrix_d;
  using stan::diff::matrix_v;
 
  matrix_d d1(3,1);
  matrix_v v1(1,3);
  
  d1 << 100, 0, -3;
  v1 << 100, 0, -3;
  
  AVAR output;
  output = min(d1);
  EXPECT_FLOAT_EQ(-3, output.val());
                   
  output = min(v1);
  EXPECT_FLOAT_EQ(-3, output.val());
}
TEST(DiffRevMatrix, min_matrix_exception) {
  using stan::math::min;
  using stan::diff::matrix_v;

  matrix_v v;
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), min(v).val());
}
