#include <stan/math/prim/mat/fun/variance.hpp>
#include <gtest/gtest.h>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <stan/math/fwd/mat/fun/typedefs.hpp>
#include <stan/math/fwd/core.hpp>

TEST(AgradFwdMatrixVariance, fd_vector) {
  using stan::math::variance;
  using stan::math::vector_d;
  using stan::math::vector_fd;

  vector_d d(1);
  d << 12.9;
  EXPECT_FLOAT_EQ(0.0,variance(d));

  vector_d d1(6);
  vector_fd v1(6);
  
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
TEST(AgradFwdMatrixVariance, fd_vector_exception) {
  using stan::math::variance;
  using stan::math::vector_d;
  using stan::math::vector_fd;

  vector_d d1;
  vector_fd v1;
  EXPECT_THROW(variance(d1), std::invalid_argument);
  EXPECT_THROW(variance(v1), std::invalid_argument);
}
TEST(AgradFwdMatrixVariance, fd_rowvector) {
  using stan::math::variance;
  using stan::math::row_vector_d;
  using stan::math::row_vector_fd;

  row_vector_d d(1);
  d << 12.9;
  EXPECT_FLOAT_EQ(0.0,variance(d));

  row_vector_d d1(6);
  row_vector_fd v1(6);
  
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
TEST(AgradFwdMatrixVariance, fd_rowvector_exception) {
  using stan::math::variance;
  using stan::math::row_vector_d;
  using stan::math::row_vector_fd;

  row_vector_d d1;
  row_vector_fd v1;
  EXPECT_THROW(variance(d1), std::invalid_argument);
  EXPECT_THROW(variance(v1), std::invalid_argument);
}
TEST(AgradFwdMatrixVariance, fd_matrix) {
  using stan::math::variance;
  using stan::math::matrix_d;
  using stan::math::matrix_fd;
  
  matrix_d m(1,1);
  m << 12.9;
  EXPECT_FLOAT_EQ(0.0,variance(m));

  matrix_d d1(2, 3);
  matrix_fd v1(2, 3);
  
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
TEST(AgradFwdMatrixVariance, fd_matrix_exception) {
  using stan::math::variance;
  using stan::math::matrix_d;
  using stan::math::matrix_fd;

  matrix_d d1;
  matrix_fd v1;
  EXPECT_THROW(variance(d1), std::invalid_argument);
  EXPECT_THROW(variance(v1), std::invalid_argument);

  d1.resize(0,1);
  v1.resize(0,1);
  EXPECT_THROW(variance(d1), std::invalid_argument);
  EXPECT_THROW(variance(v1), std::invalid_argument);

  d1.resize(1,0);
  v1.resize(1,0);
  EXPECT_THROW(variance(d1), std::invalid_argument);
  EXPECT_THROW(variance(v1), std::invalid_argument);
}
TEST(AgradFwdMatrixVariance, fd_StdVector) {
  using stan::math::variance;
  using stan::math::fvar;

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
TEST(AgradFwdMatrixVariance, ffd_vector) {
  using stan::math::variance;
  using stan::math::vector_d;
  using stan::math::vector_ffd;

  vector_d d(1);
  d << 12.9;
  EXPECT_FLOAT_EQ(0.0,variance(d));

  vector_d d1(6);
  vector_ffd v1(6);
  
  d1 << 1, 2, 3, 4, 5, 6;
  v1 << 1, 2, 3, 4, 5, 6;
   v1(0).d_ = 1.0;
   v1(1).d_ = 2.0;
   v1(2).d_ = 2.0;
   v1(3).d_ = 2.0;
   v1(4).d_ = 2.0;
   v1(5).d_ = 2.0;
  
  EXPECT_FLOAT_EQ(17.5/5.0, variance(d1));
                   
  EXPECT_FLOAT_EQ(17.5/5.0, variance(v1).val_.val());
  EXPECT_FLOAT_EQ(1.0, variance(v1).d_.val());

  d1.resize(1);
  v1.resize(1);
  EXPECT_FLOAT_EQ(0.0, variance(d1));
  EXPECT_FLOAT_EQ(0.0, variance(v1).val_.val());  
  EXPECT_FLOAT_EQ(0.0, variance(v1).d_.val());  
}
TEST(AgradFwdMatrixVariance, ffd_vector_exception) {
  using stan::math::variance;
  using stan::math::vector_d;
  using stan::math::vector_ffd;

  vector_d d1;
  vector_ffd v1;
  EXPECT_THROW(variance(d1), std::invalid_argument);
  EXPECT_THROW(variance(v1), std::invalid_argument);
}
TEST(AgradFwdMatrixVariance, ffd_rowvector) {
  using stan::math::variance;
  using stan::math::row_vector_d;
  using stan::math::row_vector_ffd;

  row_vector_d d(1);
  d << 12.9;
  EXPECT_FLOAT_EQ(0.0,variance(d));

  row_vector_d d1(6);
  row_vector_ffd v1(6);
  
  d1 << 1, 2, 3, 4, 5, 6;
  v1 << 1, 2, 3, 4, 5, 6;
   v1(0).d_ = 11.0;
   v1(1).d_ = 2.0;
   v1(2).d_ = 2.0;
   v1(3).d_ = 2.0;
   v1(4).d_ = 2.0;
   v1(5).d_ = 2.0;
  
  EXPECT_FLOAT_EQ(17.5/5.0, variance(d1));
                   
  EXPECT_FLOAT_EQ(17.5/5.0, variance(v1).val_.val());
  EXPECT_FLOAT_EQ(-9.0, variance(v1).d_.val());

  d1.resize(1);
  v1.resize(1);
  EXPECT_FLOAT_EQ(0.0, variance(d1));
  EXPECT_FLOAT_EQ(0.0, variance(v1).val_.val());  
  EXPECT_FLOAT_EQ(0.0, variance(v1).d_.val());  
}
TEST(AgradFwdMatrixVariance, ffd_rowvector_exception) {
  using stan::math::variance;
  using stan::math::row_vector_d;
  using stan::math::row_vector_ffd;

  row_vector_d d1;
  row_vector_ffd v1;
  EXPECT_THROW(variance(d1), std::invalid_argument);
  EXPECT_THROW(variance(v1), std::invalid_argument);
}
TEST(AgradFwdMatrixVariance, ffd_matrix) {
  using stan::math::variance;
  using stan::math::matrix_d;
  using stan::math::matrix_ffd;
  
  matrix_d m(1,1);
  m << 12.9;
  EXPECT_FLOAT_EQ(0.0,variance(m));

  matrix_d d1(2, 3);
  matrix_ffd v1(2, 3);
  
  d1 << 1, 2, 3, 4, 5, 6;
  v1 << 1, 2, 3, 4, 5, 6;
   v1(0,0).d_ = 1.0;
   v1(0,1).d_ = 2.0;
   v1(0,2).d_ = 2.0;
   v1(1,0).d_ = 2.0;
   v1(1,1).d_ = 2.0;
   v1(1,2).d_ = 2.0;
  
  EXPECT_FLOAT_EQ(17.5/5.0, variance(d1));
                   
  EXPECT_FLOAT_EQ(17.5/5.0, variance(v1).val_.val());
  EXPECT_FLOAT_EQ(1.0, variance(v1).d_.val());

  d1.resize(1,1);
  v1.resize(1,1);
  EXPECT_FLOAT_EQ(0.0, variance(d1));
  EXPECT_FLOAT_EQ(0.0, variance(v1).val_.val());  
  EXPECT_FLOAT_EQ(0.0, variance(v1).d_.val());  
}
TEST(AgradFwdMatrixVariance, ffd_matrix_exception) {
  using stan::math::variance;
  using stan::math::matrix_d;
  using stan::math::matrix_ffd;

  matrix_d d1;
  matrix_ffd v1;
  EXPECT_THROW(variance(d1), std::invalid_argument);
  EXPECT_THROW(variance(v1), std::invalid_argument);

  d1.resize(0,1);
  v1.resize(0,1);
  EXPECT_THROW(variance(d1), std::invalid_argument);
  EXPECT_THROW(variance(v1), std::invalid_argument);

  d1.resize(1,0);
  v1.resize(1,0);
  EXPECT_THROW(variance(d1), std::invalid_argument);
  EXPECT_THROW(variance(v1), std::invalid_argument);
}
TEST(AgradFwdMatrixVariance, ffd_StdVector) {
  using stan::math::variance;
  using stan::math::fvar;

  fvar<fvar<double> > x1 = 0.5;
  x1.d_ = 1.0;
  fvar<fvar<double> > x2 = 2.0;
  x2.d_ = 2.0;
  fvar<fvar<double> > x3 = 3.5;
  x3.d_ = 2.0;
  std::vector<fvar<fvar<double> > > y1;
  y1.push_back(x1);
  y1.push_back(x2);
  y1.push_back(x3);
  fvar<fvar<double> > f1 = variance(y1);

  EXPECT_FLOAT_EQ(2.25, f1.val_.val());
  EXPECT_FLOAT_EQ(1.5, f1.d_.val());
}
