#include <stan/math/prim/mat/fun/min.hpp>
#include <gtest/gtest.h>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <stan/math/fwd/mat/fun/typedefs.hpp>
#include <stan/math/fwd/core.hpp>

using stan::math::fvar;
TEST(AgradFwdMatrixMin, fd_vector) {
  using stan::math::min;
  using stan::math::vector_d;
  using stan::math::vector_fd;
  
  vector_d d1(3);
  vector_fd v1(3);
  
  d1 << 100, 0, -3;
  v1 << 100, 0, -3;
   v1(0).d_ = 1.0;
   v1(1).d_ = 1.0;
   v1(2).d_ = 1.0;
  
  fvar<double> output;
  output = min(d1);
  EXPECT_FLOAT_EQ(-3, output.val_);
  EXPECT_FLOAT_EQ(0, output.d_);
                   
  output = min(v1);
  EXPECT_FLOAT_EQ(-3, output.val_);
  EXPECT_FLOAT_EQ(1, output.d_);
}
TEST(AgradFwdMatrixMin, fd_vector_exception) {
  using stan::math::min;
  using stan::math::vector_d;
  using stan::math::vector_fd;

  vector_d d;
  vector_fd v;
  d.resize(0);
  v.resize(0);
  EXPECT_EQ(std::numeric_limits<double>::infinity(), min(v).val_);
  EXPECT_EQ(0, min(v).d_);
}
TEST(AgradFwdMatrixMin, fd_rowvector) {
  using stan::math::min;
  using stan::math::row_vector_d;
  using stan::math::row_vector_fd;

  row_vector_d d1(3);
  row_vector_fd v1(3);
  
  d1 << 100, 0, -3;
  v1 << 100, 0, -3;
   v1(0).d_ = 1.0;
   v1(1).d_ = 1.0;
   v1(2).d_ = 1.0;
  
  fvar<double> output;
  output = min(d1);
  EXPECT_FLOAT_EQ(-3, output.val_);
  EXPECT_FLOAT_EQ(0, output.d_);
                   
  output = min(v1);
  EXPECT_FLOAT_EQ(-3, output.val_);
  EXPECT_FLOAT_EQ(1, output.d_);
}
TEST(AgradFwdMatrixMin, fd_rowvector_exception) {
  using stan::math::min;
  using stan::math::row_vector_fd;

  row_vector_fd v;
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), min(v).val_);
  EXPECT_FLOAT_EQ(0, min(v).d_);
}
TEST(AgradFwdMatrixMin, fd_matrix) {
  using stan::math::min;
  using stan::math::matrix_d;
  using stan::math::matrix_fd;
 
  matrix_d d1(3,1);
  matrix_fd v1(1,3);
  
  d1 << 100, 0, -3;
  v1 << 100, 0, -3;
   v1(0,0).d_ = 1.0;
   v1(0,1).d_ = 1.0;
   v1(0,2).d_ = 1.0;
  
  fvar<double> output;
  output = min(d1);
  EXPECT_FLOAT_EQ(-3, output.val_);
  EXPECT_FLOAT_EQ(0, output.d_);
                   
  output = min(v1);
  EXPECT_FLOAT_EQ(-3, output.val_);
  EXPECT_FLOAT_EQ(1, output.d_);
}
TEST(AgradFwdMatrixMin, fd_matrix_exception) {
  using stan::math::min;
  using stan::math::matrix_fd;

  matrix_fd v;
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), min(v).val_);
  EXPECT_EQ(0, min(v).d_);
}
TEST(AgradFwdMatrixMin, ffd_vector) {
  using stan::math::min;
  using stan::math::vector_d;
  using stan::math::vector_ffd;
  
  vector_d d1(3);
  vector_ffd v1(3);
  fvar<fvar<double> > a,b,c;
  a.val_.val_ = 100.0;
  b.val_.val_ = 0.0;
  c.val_.val_ = -3.0;
  a.d_.val_ = 1.0;
  b.d_.val_ = 1.0;
  c.d_.val_ = 1.0;

  d1 << 100, 0, -3;
  v1 << a,b,c;
  
  fvar<fvar<double> > output;
  output = min(d1);
  EXPECT_FLOAT_EQ(-3, output.val_.val());
  EXPECT_FLOAT_EQ(0, output.d_.val());
                   
  output = min(v1);
  EXPECT_FLOAT_EQ(-3, output.val_.val());
  EXPECT_FLOAT_EQ(1, output.d_.val());
}
TEST(AgradFwdMatrixMin, ffd_vector_exception) {
  using stan::math::min;
  using stan::math::vector_d;
  using stan::math::vector_ffd;

  vector_d d;
  vector_ffd v;
  d.resize(0);
  v.resize(0);
  EXPECT_EQ(std::numeric_limits<double>::infinity(), min(v).val_.val());
  EXPECT_EQ(0, min(v).d_.val());
}
TEST(AgradFwdMatrixMin, ffd_rowvector) {
  using stan::math::min;
  using stan::math::row_vector_d;
  using stan::math::row_vector_ffd;

  row_vector_d d1(3);
  row_vector_ffd v1(3);
  fvar<fvar<double> > a,b,c;
  a.val_.val_ = 100.0;
  b.val_.val_ = 0.0;
  c.val_.val_ = -3.0;
  a.d_.val_ = 1.0;
  b.d_.val_ = 1.0;
  c.d_.val_ = 1.0;

  d1 << 100, 0, -3;
  v1 << a,b,c;
  
  fvar<fvar<double> > output;
  output = min(d1);
  EXPECT_FLOAT_EQ(-3, output.val_.val());
  EXPECT_FLOAT_EQ(0, output.d_.val());
                   
  output = min(v1);
  EXPECT_FLOAT_EQ(-3, output.val_.val());
  EXPECT_FLOAT_EQ(1, output.d_.val());
}
TEST(AgradFwdMatrixMin, ffd_rowvector_exception) {
  using stan::math::min;
  using stan::math::row_vector_ffd;

  row_vector_ffd v;
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), min(v).val_.val());
  EXPECT_FLOAT_EQ(0, min(v).d_.val());
}
TEST(AgradFwdMatrixMin, ffd_matrix) {
  using stan::math::min;
  using stan::math::matrix_d;
  using stan::math::matrix_ffd;
 
  matrix_d d1(3,1);
  matrix_ffd v1(1,3);
  fvar<fvar<double> > a,b,c;
  a.val_.val_ = 100.0;
  b.val_.val_ = 0.0;
  c.val_.val_ = -3.0;
  a.d_.val_ = 1.0;
  b.d_.val_ = 1.0;
  c.d_.val_ = 1.0;

  d1 << 100, 0, -3;
  v1 << a,b,c;

  fvar<fvar<double> > output;
  output = min(d1);
  EXPECT_FLOAT_EQ(-3, output.val_.val());
  EXPECT_FLOAT_EQ(0, output.d_.val());
                   
  output = min(v1);
  EXPECT_FLOAT_EQ(-3, output.val_.val());
  EXPECT_FLOAT_EQ(1, output.d_.val());
}
TEST(AgradFwdMatrixMin, ffd_matrix_exception) {
  using stan::math::min;
  using stan::math::matrix_ffd;

  matrix_ffd v;
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), min(v).val_.val());
  EXPECT_EQ(0, min(v).d_.val());
}
