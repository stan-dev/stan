#include <gtest/gtest.h>
#include <stan/math/fwd/mat/fun/typedefs.hpp>
#include <stan/math/prim/mat/fun/log_determinant.hpp>
#include <stan/math/fwd/mat/fun/log_determinant.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/fwd/scal/fun/abs.hpp>
#include <stan/math/fwd/scal/fun/log.hpp>

TEST(AgradFwdMatrixLogDeterminant,fd) {
  using stan::math::matrix_fd;
  using stan::math::fvar;
  using stan::math::log_determinant;
  
  matrix_fd v(2,2);
  v << 0, 1, 2, 3;
  v(0,0).d_ = 1.0;
  v(0,1).d_ = 2.0;
  v(1,0).d_ = 2.0;
  v(1,1).d_ = 2.0;
  
  fvar<double> det;
  det = log_determinant(v);
  EXPECT_FLOAT_EQ(std::log(2.0), det.val_);
  EXPECT_FLOAT_EQ(1.5, det.d_);
}

TEST(AgradFwdMatrixLogDeterminant,fd_exception) {
  using stan::math::matrix_fd;
  using stan::math::log_determinant;
  
  EXPECT_THROW(log_determinant(matrix_fd(2,3)), std::invalid_argument);
}
TEST(AgradFwdMatrixLogDeterminant,ffd) {
  using stan::math::matrix_ffd;
  using stan::math::fvar;
  using stan::math::log_determinant;
  
  fvar<fvar<double> > a,b,c,d;
  a.val_.val_ = 0.0;
  a.d_.val_ = 1.0;  
  b.val_.val_ = 1.0;
  b.d_.val_ = 2.0;
  c.val_.val_ = 2.0;
  c.d_.val_ = 2.0;
  d.val_.val_ = 3.0;
  d.d_.val_ = 2.0; 

  matrix_ffd v(2,2);
  v << a,b,c,d;
  
  fvar<fvar<double> > det;
  det = log_determinant(v);
  EXPECT_FLOAT_EQ(std::log(2.0), det.val_.val());
  EXPECT_FLOAT_EQ(1.5, det.d_.val());
}

TEST(AgradFwdMatrixLogDeterminant,ffd_exception) {
  using stan::math::matrix_ffd;
  using stan::math::log_determinant;
  
  EXPECT_THROW(log_determinant(matrix_ffd(2,3)), std::invalid_argument);
}
