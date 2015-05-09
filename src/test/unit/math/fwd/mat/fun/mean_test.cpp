#include <stan/math/prim/mat/fun/mean.hpp>
#include <gtest/gtest.h>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <stan/math/fwd/mat/fun/typedefs.hpp>
#include <stan/math/fwd/mat/fun/Eigen_NumTraits.hpp>
#include <stan/math/fwd/core.hpp>

TEST(AgradFwdMatrixMean, fd_vector) {
  using stan::math::mean;
  using stan::math::vector_d;
  using stan::math::vector_fd;
  using stan::math::fvar;

  vector_d d1(3);
  vector_fd v1(3);
  
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
TEST(AgradFwdMatrixMean, fd_vector_exception) {
  using stan::math::mean;
  using stan::math::vector_d;
  using stan::math::vector_fd;

  vector_d d;
  vector_fd v;
  EXPECT_THROW(mean(d), std::invalid_argument);
  EXPECT_THROW(mean(v), std::invalid_argument);
}
TEST(AgradFwdMatrixMean, fd_rowvector) {
  using stan::math::mean;
  using stan::math::row_vector_d;
  using stan::math::row_vector_fd;
  using stan::math::fvar;

  row_vector_d d1(3);
  row_vector_fd v1(3);
  
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
TEST(AgradFwdMatrixMean, fd_rowvector_exception) {
  using stan::math::mean;
  using stan::math::row_vector_d;
  using stan::math::row_vector_fd;

  row_vector_d d;
  row_vector_fd v;
  EXPECT_THROW(mean(d), std::invalid_argument);
  EXPECT_THROW(mean(v), std::invalid_argument);
}
TEST(AgradFwdMatrixMean, fd_matrix) {
  using stan::math::mean;
  using stan::math::matrix_d;
  using stan::math::matrix_fd;
  using stan::math::fvar;

  matrix_d d1(3,1);
  matrix_fd v1(1,3);
  
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
TEST(AgradFwdMatrixMean, fd_matrix_exception) {
  using stan::math::mean;
  using stan::math::matrix_d;
  using stan::math::matrix_fd;
 
  matrix_d d;
  matrix_fd v;
  EXPECT_THROW(mean(d), std::invalid_argument);
  EXPECT_THROW(mean(v), std::invalid_argument);
}
TEST(AgradFwdMatrixMean, fd_StdVector) {
  using stan::math::mean;
  using stan::math::fvar;

  std::vector<fvar<double> > x(0);
  EXPECT_THROW(mean(x), std::invalid_argument);
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
TEST(AgradFwdMatrixMean, ffd_vector) {
  using stan::math::mean;
  using stan::math::vector_d;
  using stan::math::vector_ffd;
  using stan::math::fvar;

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
  output = mean(d1);
  EXPECT_FLOAT_EQ(97.0/3.0, output.val_.val());
  EXPECT_FLOAT_EQ(0.0, output.d_.val());
                   
  output = mean(v1);
  EXPECT_FLOAT_EQ(97.0/3.0, output.val_.val());
  EXPECT_FLOAT_EQ(1.0, output.d_.val());
}
TEST(AgradFwdMatrixMean, ffd_vector_exception) {
  using stan::math::mean;
  using stan::math::vector_d;
  using stan::math::vector_ffd;

  vector_d d;
  vector_ffd v;
  EXPECT_THROW(mean(d), std::invalid_argument);
  EXPECT_THROW(mean(v), std::invalid_argument);
}
TEST(AgradFwdMatrixMean, ffd_rowvector) {
  using stan::math::mean;
  using stan::math::row_vector_d;
  using stan::math::row_vector_ffd;
  using stan::math::fvar;

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
  output = mean(d1);
  EXPECT_FLOAT_EQ(97.0/3.0, output.val_.val());
  EXPECT_FLOAT_EQ(0.0, output.d_.val());
                   
  output = mean(v1);
  EXPECT_FLOAT_EQ(97.0/3.0, output.val_.val());
  EXPECT_FLOAT_EQ(1.0, output.d_.val());
}
TEST(AgradFwdMatrixMean, ffd_rowvector_exception) {
  using stan::math::mean;
  using stan::math::row_vector_d;
  using stan::math::row_vector_ffd;

  row_vector_d d;
  row_vector_ffd v;
  EXPECT_THROW(mean(d), std::invalid_argument);
  EXPECT_THROW(mean(v), std::invalid_argument);
}
TEST(AgradFwdMatrixMean, ffd_matrix) {
  using stan::math::mean;
  using stan::math::matrix_d;
  using stan::math::matrix_ffd;
  using stan::math::fvar;

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
  output = mean(d1);
  EXPECT_FLOAT_EQ(97.0/3.0, output.val_.val());
  EXPECT_FLOAT_EQ(0.0, output.d_.val());
                   
  output = mean(v1);
  EXPECT_FLOAT_EQ(97.0/3.0, output.val_.val());
  EXPECT_FLOAT_EQ(1.0, output.d_.val());
}
TEST(AgradFwdMatrixMean, ffd_matrix_exception) {
  using stan::math::mean;
  using stan::math::matrix_d;
  using stan::math::matrix_ffd;
 
  matrix_d d;
  matrix_ffd v;
  EXPECT_THROW(mean(d), std::invalid_argument);
  EXPECT_THROW(mean(v), std::invalid_argument);
}
TEST(AgradFwdMatrixMean, ffd_StdVector) {
  using stan::math::mean;
  using stan::math::fvar;

  std::vector<fvar<fvar<double> > > x(0);
  EXPECT_THROW(mean(x), std::invalid_argument);
  x.push_back(1.0);
  EXPECT_FLOAT_EQ(1.0, mean(x).val_.val());
  x.push_back(2.0);
  EXPECT_FLOAT_EQ(1.5, mean(x).val_.val());

  std::vector<fvar<fvar<double> > > y;
  fvar<fvar<double> > a,b;
  a.val_.val_ = 1.0;
  a.d_.val_ = 1.0;
  b.val_.val_ = 2.0;
  b.d_.val_ = 1.0;
  y.push_back(a);
  y.push_back(b);
  fvar<fvar<double> > f = mean(y);

  EXPECT_FLOAT_EQ(1.5, f.val_.val());
  EXPECT_FLOAT_EQ(1.0, f.d_.val());
}
