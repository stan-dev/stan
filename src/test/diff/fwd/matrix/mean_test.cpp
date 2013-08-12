#include <stan/math/matrix/mean.hpp>
#include <gtest/gtest.h>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/agrad/fwd/matrix/typedefs.hpp>
#include <stan/agrad/fvar.hpp>

TEST(AgradFwdMatrix, mean_vector) {
  using stan::math::mean;
  using stan::math::vector_d;
  using stan::agrad::vector_fv;
  using stan::agrad::fvar;

  vector_d d1(3);
  vector_fv v1(3);
  
  d1 << 100, 0, -3;
  v1 << 100, 0, -3;
   v1(0).d_ = 1.0;
   v1(1).d_ = 1.0;
   v1(2).d_ = 1.0;
  
  fvar<double> output;
  output = mean(d1);
  EXPECT_FLOAT_EQ(97.0/3.0, output.val_);
  EXPECT_FLOAT_EQ(0.0, output.d_);
                   
  output = mean(v1);
  EXPECT_FLOAT_EQ(97.0/3.0, output.val_);
  EXPECT_FLOAT_EQ(1.0, output.d_);
}
TEST(AgradFwdMatrix, mean_vector_exception) {
  using stan::math::mean;
  using stan::math::vector_d;
  using stan::agrad::vector_fv;

  vector_d d;
  vector_fv v;
  EXPECT_THROW(mean(d), std::domain_error);
  EXPECT_THROW(mean(v), std::domain_error);
}
TEST(AgradFwdMatrix, mean_rowvector) {
  using stan::math::mean;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_fv;
  using stan::agrad::fvar;

  row_vector_d d1(3);
  row_vector_fv v1(3);
  
  d1 << 100, 0, -3;
  v1 << 100, 0, -3;
   v1(0).d_ = 1.0;
   v1(1).d_ = 1.0;
   v1(2).d_ = 1.0;
  
  fvar<double> output;
  output = mean(d1);
  EXPECT_FLOAT_EQ(97.0/3.0, output.val_);
  EXPECT_FLOAT_EQ(0.0, output.d_);
                   
  output = mean(v1);
  EXPECT_FLOAT_EQ(97.0/3.0, output.val_);
  EXPECT_FLOAT_EQ(1.0, output.d_);
}
TEST(AgradFwdMatrix, mean_rowvector_exception) {
  using stan::math::mean;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_fv;

  row_vector_d d;
  row_vector_fv v;
  EXPECT_THROW(mean(d), std::domain_error);
  EXPECT_THROW(mean(v), std::domain_error);
}
TEST(AgradFwdMatrix, mean_matrix) {
  using stan::math::mean;
  using stan::math::matrix_d;
  using stan::agrad::matrix_fv;
  using stan::agrad::fvar;

  matrix_d d1(3,1);
  matrix_fv v1(1,3);
  
  d1 << 100, 0, -3;
  v1 << 100, 0, -3;
   v1(0,0).d_ = 1.0;
   v1(0,1).d_ = 1.0;
   v1(0,2).d_ = 1.0;
  
  fvar<double> output;
  output = mean(d1);
  EXPECT_FLOAT_EQ(97.0/3.0, output.val_);
  EXPECT_FLOAT_EQ(0.0, output.d_);
                   
  output = mean(v1);
  EXPECT_FLOAT_EQ(97.0/3.0, output.val_);
  EXPECT_FLOAT_EQ(1.0, output.d_);
}
TEST(AgradFwdMatrix, mean_matrix_exception) {
  using stan::math::mean;
  using stan::math::matrix_d;
  using stan::agrad::matrix_fv;
 
  matrix_d d;
  matrix_fv v;
  EXPECT_THROW(mean(d), std::domain_error);
  EXPECT_THROW(mean(v), std::domain_error);
}
TEST(AgradFwdMatrix, meanStdVector) {
  using stan::math::mean;
  using stan::agrad::fvar;

  std::vector<fvar<double> > x(0);
  EXPECT_THROW(mean(x), std::domain_error);
  x.push_back(1.0);
  EXPECT_FLOAT_EQ(1.0, mean(x).val_);
  x.push_back(2.0);
  EXPECT_FLOAT_EQ(1.5, mean(x).val_);

  std::vector<fvar<double> > y;
  fvar<double> a = 1.0;
  a.d_ = 1.0;
  fvar<double> b = 2.0;
  b.d_ = 1.0;
  y.push_back(a);
  y.push_back(b);
  fvar<double> f = mean(y);

  EXPECT_FLOAT_EQ(1.5, f.val_);
  EXPECT_FLOAT_EQ(1.0, f.d_);
}
