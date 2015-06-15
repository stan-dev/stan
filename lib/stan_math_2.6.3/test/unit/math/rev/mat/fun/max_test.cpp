#include <stan/math/prim/mat/meta/get.hpp>
#include <stan/math/prim/arr/meta/get.hpp>
#include <stan/math/prim/mat/meta/length.hpp>
#include <stan/math/prim/mat/meta/is_vector.hpp>
#include <stan/math/prim/mat/meta/is_vector_like.hpp>
#include <stan/math/prim/mat/fun/max.hpp>
#include <gtest/gtest.h>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <stan/math/rev/mat/fun/typedefs.hpp>
#include <stan/math/rev/core.hpp>

TEST(AgradRevMatrix, max_vector) {
  using stan::math::max;
  using stan::math::vector_d;
  using stan::math::vector_v;

  vector_d d1(3);
  vector_v v1(3);
  
  d1 << 100, 0, -3;
  v1 << 100, 0, -3;
  
  AVAR output;
  output = max(d1);
  EXPECT_FLOAT_EQ(100, output.val());
                   
  output = max(v1);
  EXPECT_FLOAT_EQ(100, output.val());
}
TEST(AgradRevMatrix, max_vector_exception) {
  using stan::math::max;
  using stan::math::vector_v;

  vector_v v;
  EXPECT_EQ(-std::numeric_limits<double>::infinity(), max(v).val());
}
TEST(AgradRevMatrix, max_rowvector) {
  using stan::math::max;
  using stan::math::row_vector_d;
  using stan::math::row_vector_v;

  row_vector_d d1(3);
  row_vector_v v1(3);
  
  d1 << 100, 0, -3;
  v1 << 100, 0, -3;
  
  AVAR output;
  output = max(d1);
  EXPECT_FLOAT_EQ(100, output.val());
                   
  output = max(v1);
  EXPECT_FLOAT_EQ(100, output.val());
}
TEST(AgradRevMatrix, max_rowvector_exception) {
  using stan::math::max;
  using stan::math::row_vector_v;

  row_vector_v v;
  EXPECT_EQ(-std::numeric_limits<double>::infinity(), max(v).val());
}
TEST(AgradRevMatrix, max_matrix) {
  using stan::math::max;
  using stan::math::matrix_d;
  using stan::math::matrix_v;

  matrix_d d1(3,1);
  matrix_v v1(1,3);
  
  d1 << 100, 0, -3;
  v1 << 100, 0, -3;
  
  AVAR output;
  output = max(d1);
  EXPECT_FLOAT_EQ(100, output.val());
                   
  output = max(v1);
  EXPECT_FLOAT_EQ(100, output.val());
}
TEST(AgradRevMatrix, max_matrix_exception) {
  using stan::math::max;
  using stan::math::matrix_v;
  
  matrix_v v;
  EXPECT_EQ(-std::numeric_limits<double>::infinity(), max(v).val());
}
