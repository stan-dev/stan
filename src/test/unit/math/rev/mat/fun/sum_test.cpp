#include <stan/math/prim/mat/meta/get.hpp>
#include <stan/math/prim/arr/meta/get.hpp>
#include <stan/math/prim/mat/meta/length.hpp>
#include <stan/math/prim/mat/meta/is_vector.hpp>
#include <stan/math/prim/mat/meta/is_vector_like.hpp>
#include <stan/math/rev/mat/fun/sum.hpp>
#include <gtest/gtest.h>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <stan/math/prim/mat/fun/sum.hpp>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <stan/math/rev/mat/fun/typedefs.hpp>

TEST(AgradRevMatrix, sum_vector) {
  using stan::math::sum;
  using stan::math::vector_d;
  using stan::math::vector_v;

  vector_d d(6);
  vector_v v(6);
  
  d << 1, 2, 3, 4, 5, 6;
  v << 1, 2, 3, 4, 5, 6;
  
  AVAR output;
  output = sum(d);
  EXPECT_FLOAT_EQ(21.0, output.val());
  
  output = sum(v);
  EXPECT_FLOAT_EQ(21.0, output.val());

  std::vector<double> grad;
  std::vector<AVAR> x(v.size());
  for (int i = 0; i < v.size(); ++i)
    x[i] = v(i);
  output.grad(x, grad);
  EXPECT_EQ(6, grad.size());
  for (int i = 0; i < 6; ++i)
    EXPECT_FLOAT_EQ(1.0, grad[i]);
                   
  d.resize(0);
  v.resize(0);
  EXPECT_FLOAT_EQ(0.0, sum(d));
  EXPECT_FLOAT_EQ(0.0, sum(v).val());
}
TEST(AgradRevMatrix, sum_rowvector) {
  using stan::math::sum;
  using stan::math::row_vector_d;
  using stan::math::row_vector_v;

  row_vector_d d(6);
  row_vector_v v(6);
  
  d << 1, 2, 3, 4, 5, 6;
  v << 1, 2, 3, 4, 5, 6;
  
  AVAR output;
  output = sum(d);
  EXPECT_FLOAT_EQ(21.0, output.val());
                   
  output = sum(v);
  EXPECT_FLOAT_EQ(21.0, output.val());

  d.resize(0);
  v.resize(0);
  EXPECT_FLOAT_EQ(0.0, sum(d));
  EXPECT_FLOAT_EQ(0.0, sum(v).val());
}
TEST(AgradRevMatrix, sum_matrix) {
  using stan::math::sum;
  using stan::math::matrix_d;
  using stan::math::matrix_v;

  matrix_d d(2, 3);
  matrix_v v(2, 3);
  
  d << 1, 2, 3, 4, 5, 6;
  v << 1, 2, 3, 4, 5, 6;
  
  AVAR output;
  output = sum(d);
  EXPECT_FLOAT_EQ(21.0, output.val());
                   
  output = sum(v);
  EXPECT_FLOAT_EQ(21.0, output.val());

  d.resize(0, 0);
  v.resize(0, 0);
  EXPECT_FLOAT_EQ(0.0, sum(d));
  EXPECT_FLOAT_EQ(0.0, sum(v).val());
}
