#include <stan/math/prim/mat/fun/sd.hpp>
#include <gtest/gtest.h>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <stan/math/fwd/mat/fun/typedefs.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/fwd/scal/fun/sqrt.hpp>

TEST(AgradFwdMatrixSD, fd_vector) {
  using stan::math::sd;
  using stan::math::vector_d;
  using stan::math::vector_fd;

  vector_d v(1);
  v << 1.0;
  EXPECT_FLOAT_EQ(0.0, sd(v));

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
  
  EXPECT_FLOAT_EQ(std::sqrt(17.5/5.0), sd(d1));
                   
  EXPECT_FLOAT_EQ(std::sqrt(17.5/5.0), sd(v1).val_);
  EXPECT_FLOAT_EQ(std::sqrt(1.0/14.0), sd(v1).d_);
  
  d1.resize(1);
  v1.resize(1);
  EXPECT_FLOAT_EQ(0.0, sd(d1));
  EXPECT_FLOAT_EQ(0.0, sd(v1).val_);
  EXPECT_FLOAT_EQ(0.0, sd(v1).d_);
}
TEST(AgradFwdMatrixSD, fd_vector_exception) {
  using stan::math::sd;
  using stan::math::vector_d;
  using stan::math::vector_fd;

  vector_d d1;
  vector_fd v1;
  EXPECT_THROW(sd(d1), std::invalid_argument);
  EXPECT_THROW(sd(v1), std::invalid_argument);
}
TEST(AgradFwdMatrixSD, fd_rowvector) {
  using stan::math::sd;
  using stan::math::row_vector_d;
  using stan::math::row_vector_fd;

  row_vector_d v(1);
  v << 1.0;
  EXPECT_FLOAT_EQ(0.0, sd(v));


  row_vector_d d1(6);
  row_vector_fd v1(6);
  
  d1 << 1, 2, 3, 4, 5, 6;
  v1 << 1, 2, 3, 4, 5, 6;
   v1(0).d_ = 1.0;
   v1(1).d_ = 2.0;
   v1(2).d_ = 2.0;
   v1(3).d_ = 2.0;
   v1(4).d_ = 2.0;
   v1(5).d_ = 2.0;
  
  EXPECT_FLOAT_EQ(std::sqrt(17.5/5.0), sd(d1));
                   
  EXPECT_FLOAT_EQ(std::sqrt(17.5/5.0), sd(v1).val_);
  EXPECT_FLOAT_EQ(std::sqrt(1.0/14.0), sd(v1).d_);

  d1.resize(1);
  v1.resize(1);
  EXPECT_FLOAT_EQ(0.0, sd(d1));
  EXPECT_FLOAT_EQ(0.0, sd(v1).val_);
  EXPECT_FLOAT_EQ(0.0, sd(v1).d_);
}
TEST(AgradFwdMatrixSD, fd_rowvector_exception) {
  using stan::math::sd;
  using stan::math::row_vector_d;
  using stan::math::row_vector_fd;

  row_vector_d d;
  row_vector_fd v;
  
  EXPECT_THROW(sd(d), std::invalid_argument);
  EXPECT_THROW(sd(v), std::invalid_argument);
}
TEST(AgradFwdMatrixSD, fd_matrix) {
  using stan::math::sd;
  using stan::math::matrix_d;
  using stan::math::matrix_fd;

  matrix_d v(1,1);
  v << 1.0;
  EXPECT_FLOAT_EQ(0.0, sd(v));

  matrix_d d1(2, 3);
  matrix_fd v1(2, 3);
  
  d1 << 1, 2, 3, 4, 5, 6;
  v1 << 1, 2, 3, 4, 5, 6;
   v1(0).d_ = 1.0;
   v1(1).d_ = 2.0;
   v1(2).d_ = 2.0;
   v1(3).d_ = 2.0;
   v1(4).d_ = 2.0;
   v1(5).d_ = 2.0;
  
  EXPECT_FLOAT_EQ(std::sqrt(17.5/5.0), sd(d1));
  EXPECT_FLOAT_EQ(std::sqrt(17.5/5.0), sd(v1).val_);
  EXPECT_FLOAT_EQ(std::sqrt(1.0/14.0), sd(v1).d_);

  d1.resize(1, 1);
  v1.resize(1, 1);
  EXPECT_FLOAT_EQ(0.0, sd(d1));
  EXPECT_FLOAT_EQ(0.0, sd(v1).val_);
  EXPECT_FLOAT_EQ(0.0, sd(v1).d_);
}
TEST(AgradFwdMatrixSD, fd_matrix_exception) {
  using stan::math::sd;
  using stan::math::matrix_d;
  using stan::math::matrix_fd;

  matrix_d d;
  matrix_fd v;

  EXPECT_THROW(sd(d), std::invalid_argument);
  EXPECT_THROW(sd(v), std::invalid_argument);

  d.resize(1, 0);
  v.resize(1, 0);
  EXPECT_THROW(sd(d), std::invalid_argument);
  EXPECT_THROW(sd(v), std::invalid_argument);

  d.resize(0, 1);
  v.resize(0, 1);
  EXPECT_THROW(sd(d), std::invalid_argument);
  EXPECT_THROW(sd(v), std::invalid_argument);
}
TEST(AgradFwdMatrixSD, ffd_vector) {
  using stan::math::sd;
  using stan::math::vector_d;
  using stan::math::vector_ffd;

  vector_d v(1);
  v << 1.0;
  EXPECT_FLOAT_EQ(0.0, sd(v));

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
  
  EXPECT_FLOAT_EQ(std::sqrt(17.5/5.0), sd(d1));
                   
  EXPECT_FLOAT_EQ(std::sqrt(17.5/5.0), sd(v1).val_.val());
  EXPECT_FLOAT_EQ(0.26726124, sd(v1).d_.val());
  
  d1.resize(1);
  v1.resize(1);
  EXPECT_FLOAT_EQ(0.0, sd(d1));
  EXPECT_FLOAT_EQ(0.0, sd(v1).val_.val());
  EXPECT_FLOAT_EQ(0.0, sd(v1).d_.val());
}
TEST(AgradFwdMatrixSD, ffd_vector_exception) {
  using stan::math::sd;
  using stan::math::vector_d;
  using stan::math::vector_ffd;

  vector_d d1;
  vector_ffd v1;
  EXPECT_THROW(sd(d1), std::invalid_argument);
  EXPECT_THROW(sd(v1), std::invalid_argument);
}
TEST(AgradFwdMatrixSD, ffd_rowvector) {
  using stan::math::sd;
  using stan::math::row_vector_d;
  using stan::math::row_vector_ffd;

  row_vector_d v(1);
  v << 1.0;
  EXPECT_FLOAT_EQ(0.0, sd(v));


  row_vector_d d1(6);
  row_vector_ffd v1(6);
  
  d1 << 1, 2, 3, 4, 5, 6;
  v1 << 1, 2, 3, 4, 5, 6;
   v1(0).d_ = 1.0;
   v1(1).d_ = 2.0;
   v1(2).d_ = 2.0;
   v1(3).d_ = 2.0;
   v1(4).d_ = 2.0;
   v1(5).d_ = 2.0;
  
  EXPECT_FLOAT_EQ(std::sqrt(17.5/5.0), sd(d1));
                   
  EXPECT_FLOAT_EQ(std::sqrt(17.5/5.0), sd(v1).val_.val());
  EXPECT_FLOAT_EQ(0.26726124, sd(v1).d_.val());

  d1.resize(1);
  v1.resize(1);
  EXPECT_FLOAT_EQ(0.0, sd(d1));
  EXPECT_FLOAT_EQ(0.0, sd(v1).val_.val());
  EXPECT_FLOAT_EQ(0.0, sd(v1).d_.val());
}
TEST(AgradFwdMatrixSD, ffd_rowvector_exception) {
  using stan::math::sd;
  using stan::math::row_vector_d;
  using stan::math::row_vector_ffd;

  row_vector_d d;
  row_vector_ffd v;
  
  EXPECT_THROW(sd(d), std::invalid_argument);
  EXPECT_THROW(sd(v), std::invalid_argument);
}
TEST(AgradFwdMatrixSD, ffd_matrix) {
  using stan::math::sd;
  using stan::math::matrix_d;
  using stan::math::matrix_ffd;

  matrix_d v(1,1);
  v << 1.0;
  EXPECT_FLOAT_EQ(0.0, sd(v));

  matrix_d d1(2, 3);
  matrix_ffd v1(2, 3);
  
  d1 << 1, 2, 3, 4, 5, 6;
  v1 << 1, 2, 3, 4, 5, 6;
   v1(0).d_ = 1.0;
   v1(1).d_ = 2.0;
   v1(2).d_ = 2.0;
   v1(3).d_ = 2.0;
   v1(4).d_ = 2.0;
   v1(5).d_ = 2.0;
  
  EXPECT_FLOAT_EQ(std::sqrt(17.5/5.0), sd(d1));
  EXPECT_FLOAT_EQ(std::sqrt(17.5/5.0), sd(v1).val_.val());
  EXPECT_FLOAT_EQ(0.26726124, sd(v1).d_.val());

  d1.resize(1, 1);
  v1.resize(1, 1);
  EXPECT_FLOAT_EQ(0.0, sd(d1));
  EXPECT_FLOAT_EQ(0.0, sd(v1).val_.val());
  EXPECT_FLOAT_EQ(0.0, sd(v1).d_.val());
}
TEST(AgradFwdMatrixSD, ffd_matrix_exception) {
  using stan::math::sd;
  using stan::math::matrix_d;
  using stan::math::matrix_ffd;

  matrix_d d;
  matrix_ffd v;

  EXPECT_THROW(sd(d), std::invalid_argument);
  EXPECT_THROW(sd(v), std::invalid_argument);

  d.resize(1, 0);
  v.resize(1, 0);
  EXPECT_THROW(sd(d), std::invalid_argument);
  EXPECT_THROW(sd(v), std::invalid_argument);

  d.resize(0, 1);
  v.resize(0, 1);
  EXPECT_THROW(sd(d), std::invalid_argument);
  EXPECT_THROW(sd(v), std::invalid_argument);
}
