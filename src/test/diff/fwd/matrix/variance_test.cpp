#include <stan/math/matrix/variance.hpp>
#include <gtest/gtest.h>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/agrad/fwd/matrix/typedefs.hpp>
#include <stan/agrad/fvar.hpp>

TEST(AgradFwdMatrix, variance_vector) {
  using stan::math::variance;
  using stan::math::vector_d;
  using stan::agrad::vector_fv;

  vector_d d(1);
  d << 12.9;
  EXPECT_FLOAT_EQ(0.0,variance(d));

  vector_d d1(6);
  vector_fv v1(6);
  
  d1 << 1, 2, 3, 4, 5, 6;
  v1 << 1, 2, 3, 4, 5, 6;
   v1(0).d_ = 1.0;
   v1(1).d_ = 2.0;
   v1(2).d_ = 2.0;
   v1(3).d_ = 2.0;
   v1(4).d_ = 2.0;
   v1(5).d_ = 2.0;
  
  EXPECT_FLOAT_EQ(17.5/5.0, variance(d1));
                   
  EXPECT_FLOAT_EQ(17.5/5.0, variance(v1).val_);
  EXPECT_FLOAT_EQ(1.0, variance(v1).d_);

  d1.resize(1);
  v1.resize(1);
  EXPECT_FLOAT_EQ(0.0, variance(d1));
  EXPECT_FLOAT_EQ(0.0, variance(v1).val_);  
  EXPECT_FLOAT_EQ(0.0, variance(v1).d_);  
}
TEST(AgradFwdMatrix, variance_vector_exception) {
  using stan::math::variance;
  using stan::math::vector_d;
  using stan::agrad::vector_fv;

  vector_d d1;
  vector_fv v1;
  EXPECT_THROW(variance(d1), std::domain_error);
  EXPECT_THROW(variance(v1), std::domain_error);
}
TEST(AgradFwdMatrix, variance_rowvector) {
  using stan::math::variance;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_fv;

  row_vector_d d(1);
  d << 12.9;
  EXPECT_FLOAT_EQ(0.0,variance(d));

  row_vector_d d1(6);
  row_vector_fv v1(6);
  
  d1 << 1, 2, 3, 4, 5, 6;
  v1 << 1, 2, 3, 4, 5, 6;
   v1(0).d_ = 11.0;
   v1(1).d_ = 2.0;
   v1(2).d_ = 2.0;
   v1(3).d_ = 2.0;
   v1(4).d_ = 2.0;
   v1(5).d_ = 2.0;
  
  EXPECT_FLOAT_EQ(17.5/5.0, variance(d1));
                   
  EXPECT_FLOAT_EQ(17.5/5.0, variance(v1).val_);
  EXPECT_FLOAT_EQ(-9.0, variance(v1).d_);

  d1.resize(1);
  v1.resize(1);
  EXPECT_FLOAT_EQ(0.0, variance(d1));
  EXPECT_FLOAT_EQ(0.0, variance(v1).val_);  
  EXPECT_FLOAT_EQ(0.0, variance(v1).d_);  
}
TEST(AgradFwdMatrix, variance_rowvector_exception) {
  using stan::math::variance;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_fv;

  row_vector_d d1;
  row_vector_fv v1;
  EXPECT_THROW(variance(d1), std::domain_error);
  EXPECT_THROW(variance(v1), std::domain_error);
}
TEST(AgradFwdMatrix, variance_matrix) {
  using stan::math::variance;
  using stan::math::matrix_d;
  using stan::agrad::matrix_fv;
  
  matrix_d m(1,1);
  m << 12.9;
  EXPECT_FLOAT_EQ(0.0,variance(m));

  matrix_d d1(2, 3);
  matrix_fv v1(2, 3);
  
  d1 << 1, 2, 3, 4, 5, 6;
  v1 << 1, 2, 3, 4, 5, 6;
   v1(0,0).d_ = 1.0;
   v1(0,1).d_ = 2.0;
   v1(0,2).d_ = 2.0;
   v1(1,0).d_ = 2.0;
   v1(1,1).d_ = 2.0;
   v1(1,2).d_ = 2.0;
  
  EXPECT_FLOAT_EQ(17.5/5.0, variance(d1));
                   
  EXPECT_FLOAT_EQ(17.5/5.0, variance(v1).val_);
  EXPECT_FLOAT_EQ(1.0, variance(v1).d_);

  d1.resize(1,1);
  v1.resize(1,1);
  EXPECT_FLOAT_EQ(0.0, variance(d1));
  EXPECT_FLOAT_EQ(0.0, variance(v1).val_);  
  EXPECT_FLOAT_EQ(0.0, variance(v1).d_);  
}
TEST(AgradFwdMatrix, variance_matrix_exception) {
  using stan::math::variance;
  using stan::math::matrix_d;
  using stan::agrad::matrix_fv;

  matrix_d d1;
  matrix_fv v1;
  EXPECT_THROW(variance(d1), std::domain_error);
  EXPECT_THROW(variance(v1), std::domain_error);

  d1.resize(0,1);
  v1.resize(0,1);
  EXPECT_THROW(variance(d1), std::domain_error);
  EXPECT_THROW(variance(v1), std::domain_error);

  d1.resize(1,0);
  v1.resize(1,0);
  EXPECT_THROW(variance(d1), std::domain_error);
  EXPECT_THROW(variance(v1), std::domain_error);
}
TEST(AgradFwdMatrix, varianceStdVector) {
  using stan::math::variance;
  using stan::agrad::fvar;

  fvar<double> x1 = 0.5;
  x1.d_ = 1.0;
  fvar<double> x2 = 2.0;
  x2.d_ = 2.0;
  fvar<double> x3 = 3.5;
  x3.d_ = 2.0;
  std::vector<fvar<double> > y1;
  y1.push_back(x1);
  y1.push_back(x2);
  y1.push_back(x3);
  fvar<double> f1 = variance(y1);

  EXPECT_FLOAT_EQ(2.25, f1.val_);
  EXPECT_FLOAT_EQ(1.5, f1.d_);
}
