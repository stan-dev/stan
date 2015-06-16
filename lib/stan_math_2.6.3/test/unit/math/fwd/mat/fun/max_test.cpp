#include <stan/math/prim/mat/fun/max.hpp>
#include <gtest/gtest.h>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <stan/math/fwd/mat/fun/typedefs.hpp>
#include <stan/math/fwd/core.hpp>

using stan::math::fvar;
TEST(AgradFwdMatrixMax, fd_vector) {
  using stan::math::max;
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
  output = max(d1);
  EXPECT_FLOAT_EQ(100, output.val_);
  EXPECT_FLOAT_EQ(0, output.d_);
                   
  output = max(v1);
  EXPECT_FLOAT_EQ(100, output.val_);
  EXPECT_FLOAT_EQ(1, output.d_);
}
TEST(AgradFwdMatrixMax, fd_vector_exception) {
  using stan::math::max;
  using stan::math::vector_fd;

  vector_fd v;
  EXPECT_EQ(-std::numeric_limits<double>::infinity(), max(v).val_);
  EXPECT_EQ(0, max(v).d_);
}
TEST(AgradFwdMatrixMax, fd_rowvector) {
  using stan::math::max;
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
  output = max(d1);
  EXPECT_FLOAT_EQ(100, output.val_);
  EXPECT_FLOAT_EQ(0, output.d_);
                   
  output = max(v1);
  EXPECT_FLOAT_EQ(100, output.val_);
  EXPECT_FLOAT_EQ(1, output.d_);
}
TEST(AgradFwdMatrixMax, fd_rowvector_exception) {
  using stan::math::max;
  using stan::math::row_vector_fd;

  row_vector_fd v;
  EXPECT_EQ(-std::numeric_limits<double>::infinity(), max(v).val_);
  EXPECT_EQ(0, max(v).d_);
}
TEST(AgradFwdMatrixMax, fd_matrix) {
  using stan::math::max;
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
  output = max(d1);
  EXPECT_FLOAT_EQ(100, output.val_);
  EXPECT_FLOAT_EQ(0, output.d_);
                   
  output = max(v1);
  EXPECT_FLOAT_EQ(100, output.val_);
  EXPECT_FLOAT_EQ(1, output.d_);
}
TEST(AgradFwdMatrixMax, fd_matrix_exception) {
  using stan::math::max;
  using stan::math::matrix_fd;
  
  matrix_fd v;
  EXPECT_EQ(-std::numeric_limits<double>::infinity(), max(v).val_);
  EXPECT_EQ(0, max(v).d_);
}
TEST(AgradFwdMatrixMax, ffd_vector) {
  using stan::math::max;
  using stan::math::vector_d;
  using stan::math::vector_ffd;
  using stan::math::fvar;

  vector_d d1(3);
  vector_ffd v1(3);
  
  fvar<fvar<double> > a,b,c;
  a.val_.val_ = 100.0;
  a.d_.val_ = 1.0;  
  b.val_.val_ = 0.0;
  b.d_.val_ = 1.0;
  c.val_.val_ = -3.0;
  c.d_.val_ = 1.0;

  d1 << 100, 0, -3;
  v1 << a,b,c;
  
  fvar<fvar<double> > output;
  output = max(d1);
  EXPECT_FLOAT_EQ(100, output.val_.val());
  EXPECT_FLOAT_EQ(0, output.d_.val());
                   
  output = max(v1);
  EXPECT_FLOAT_EQ(100, output.val_.val());
  EXPECT_FLOAT_EQ(1, output.d_.val());
}
TEST(AgradFwdMatrixMax, ffd_vector_exception) {
  using stan::math::max;
  using stan::math::vector_ffd;

  vector_ffd v;
  EXPECT_EQ(-std::numeric_limits<double>::infinity(), max(v).val_.val());
  EXPECT_EQ(0, max(v).d_.val());
}
TEST(AgradFwdMatrixMax, ffd_rowvector) {
  using stan::math::max;
  using stan::math::row_vector_d;
  using stan::math::row_vector_ffd;
  using stan::math::fvar;

  row_vector_d d1(3);
  row_vector_ffd v1(3);
  
  fvar<fvar<double> > a,b,c;
  a.val_.val_ = 100.0;
  a.d_.val_ = 1.0;  
  b.val_.val_ = 0.0;
  b.d_.val_ = 1.0;
  c.val_.val_ = -3.0;
  c.d_.val_ = 1.0;

  d1 << 100, 0, -3;
  v1 << a,b,c;
  
  fvar<fvar<double> > output;
  output = max(d1);
  EXPECT_FLOAT_EQ(100, output.val_.val());
  EXPECT_FLOAT_EQ(0, output.d_.val());
                   
  output = max(v1);
  EXPECT_FLOAT_EQ(100, output.val_.val());
  EXPECT_FLOAT_EQ(1, output.d_.val());
}
TEST(AgradFwdMatrixMax, ffd_rowvector_exception) {
  using stan::math::max;
  using stan::math::row_vector_ffd;

  row_vector_ffd v;
  EXPECT_EQ(-std::numeric_limits<double>::infinity(), max(v).val_.val());
  EXPECT_EQ(0, max(v).d_.val());
}
TEST(AgradFwdMatrixMax, ffd_matrix) {
  using stan::math::max;
  using stan::math::matrix_d;
  using stan::math::matrix_ffd;
  using stan::math::fvar;

  matrix_d d1(3,1);
  matrix_ffd v1(1,3);
  
  fvar<fvar<double> > a,b,c;
  a.val_.val_ = 100.0;
  a.d_.val_ = 1.0;  
  b.val_.val_ = 0.0;
  b.d_.val_ = 1.0;
  c.val_.val_ = -3.0;
  c.d_.val_ = 1.0;

  d1 << 100, 0, -3;
  v1 << a,b,c;
  
  fvar<fvar<double> > output;
  output = max(d1);
  EXPECT_FLOAT_EQ(100, output.val_.val());
  EXPECT_FLOAT_EQ(0, output.d_.val());
                   
  output = max(v1);
  EXPECT_FLOAT_EQ(100, output.val_.val());
  EXPECT_FLOAT_EQ(1, output.d_.val());
}
TEST(AgradFwdMatrixMax, ffd_matrix_exception) {
  using stan::math::max;
  using stan::math::matrix_ffd;
  
  matrix_ffd v;
  EXPECT_EQ(-std::numeric_limits<double>::infinity(), max(v).val_.val());
  EXPECT_EQ(0, max(v).d_.val());
}
