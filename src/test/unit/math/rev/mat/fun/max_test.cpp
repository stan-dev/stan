#include <stan/math/prim/mat/fun/max.hpp>
#include <gtest/gtest.h>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <stan/math/rev/mat/fun/typedefs.hpp>
#include <stan/math/rev/scal/fun/operator_addition.hpp>
#include <stan/math/rev/scal/fun/operator_divide_equal.hpp>
#include <stan/math/rev/scal/fun/operator_division.hpp>
#include <stan/math/rev/scal/fun/operator_equal.hpp>
#include <stan/math/rev/scal/fun/operator_greater_than.hpp>
#include <stan/math/rev/scal/fun/operator_greater_than_or_equal.hpp>
#include <stan/math/rev/scal/fun/operator_less_than.hpp>
#include <stan/math/rev/scal/fun/operator_less_than_or_equal.hpp>
#include <stan/math/rev/scal/fun/operator_minus_equal.hpp>
#include <stan/math/rev/scal/fun/operator_multiplication.hpp>
#include <stan/math/rev/scal/fun/operator_multiply_equal.hpp>
#include <stan/math/rev/scal/fun/operator_not_equal.hpp>
#include <stan/math/rev/scal/fun/operator_plus_equal.hpp>
#include <stan/math/rev/scal/fun/operator_subtraction.hpp>
#include <stan/math/rev/scal/fun/operator_unary_decrement.hpp>
#include <stan/math/rev/scal/fun/operator_unary_increment.hpp>
#include <stan/math/rev/scal/fun/operator_unary_negative.hpp>
#include <stan/math/rev/scal/fun/operator_unary_not.hpp>
#include <stan/math/rev/scal/fun/operator_unary_plus.hpp>

TEST(AgradRevMatrix, max_vector) {
  using stan::math::max;
  using stan::math::vector_d;
  using stan::agrad::vector_v;

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
  using stan::agrad::vector_v;

  vector_v v;
  EXPECT_EQ(-std::numeric_limits<double>::infinity(), max(v).val());
}
TEST(AgradRevMatrix, max_rowvector) {
  using stan::math::max;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_v;

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
  using stan::agrad::row_vector_v;

  row_vector_v v;
  EXPECT_EQ(-std::numeric_limits<double>::infinity(), max(v).val());
}
TEST(AgradRevMatrix, max_matrix) {
  using stan::math::max;
  using stan::math::matrix_d;
  using stan::agrad::matrix_v;

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
  using stan::agrad::matrix_v;
  
  matrix_v v;
  EXPECT_EQ(-std::numeric_limits<double>::infinity(), max(v).val());
}
