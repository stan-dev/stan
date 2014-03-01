#include <stan/math/matrix/mean.hpp>
#include <gtest/gtest.h>
#include <test/unit/agrad/util.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/agrad/rev/matrix/typedefs.hpp>
#include <stan/agrad/rev.hpp>

TEST(AgradRevMatrix, mean_vector) {
  using stan::math::mean;
  using stan::math::vector_d;
  using stan::agrad::vector_v;

  vector_d d1(3);
  vector_v v1(3);
  
  d1 << 100, 0, -3;
  v1 << 100, 0, -3;
  
  AVAR output;
  output = mean(d1);
  EXPECT_FLOAT_EQ(97.0/3.0, output.val());
                   
  output = mean(v1);
  EXPECT_FLOAT_EQ(97.0/3.0, output.val());
}
TEST(AgradRevMatrix, mean_vector_exception) {
  using stan::math::mean;
  using stan::math::vector_d;
  using stan::agrad::vector_v;

  vector_d d;
  vector_v v;
  EXPECT_THROW(mean(d), std::domain_error);
  EXPECT_THROW(mean(v), std::domain_error);
}
TEST(AgradRevMatrix, mean_rowvector) {
  using stan::math::mean;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_v;

  row_vector_d d1(3);
  row_vector_v v1(3);
  
  d1 << 100, 0, -3;
  v1 << 100, 0, -3;
  
  AVAR output;
  output = mean(d1);
  EXPECT_FLOAT_EQ(97.0/3.0, output.val());
                   
  output = mean(v1);
  EXPECT_FLOAT_EQ(97.0/3.0, output.val());
}
TEST(AgradRevMatrix, mean_rowvector_exception) {
  using stan::math::mean;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_v;

  row_vector_d d;
  row_vector_v v;
  EXPECT_THROW(mean(d), std::domain_error);
  EXPECT_THROW(mean(v), std::domain_error);
}
TEST(AgradRevMatrix, mean_matrix) {
  using stan::math::mean;
  using stan::math::matrix_d;
  using stan::agrad::matrix_v;

  matrix_d d1(3,1);
  matrix_v v1(1,3);
  
  d1 << 100, 0, -3;
  v1 << 100, 0, -3;
  
  AVAR output;
  output = mean(d1);
  EXPECT_FLOAT_EQ(97.0/3.0, output.val());
                   
  output = mean(v1);
  EXPECT_FLOAT_EQ(97.0/3.0, output.val());
}
TEST(AgradRevMatrix, mean_matrix_exception) {
  using stan::math::mean;
  using stan::math::matrix_d;
  using stan::agrad::matrix_v;
 
  matrix_d d;
  matrix_v v;
  EXPECT_THROW(mean(d), std::domain_error);
  EXPECT_THROW(mean(v), std::domain_error);
}
TEST(AgradRevMatrix, meanStdVector) {
  using stan::math::mean; // should use arg-dep lookup
  AVEC x(0);
  EXPECT_THROW(mean(x), std::domain_error);
  x.push_back(1.0);
  EXPECT_FLOAT_EQ(1.0, mean(x).val());
  x.push_back(2.0);
  EXPECT_FLOAT_EQ(1.5, mean(x).val());

  AVEC y = createAVEC(1.0,2.0);
  AVAR f = mean(y);
  VEC grad = cgrad(f, y[0], y[1]);
  EXPECT_FLOAT_EQ(0.5, grad[0]);
  EXPECT_FLOAT_EQ(0.5, grad[1]);
  EXPECT_EQ(2U, grad.size());
}
