#include <stan/math/prim/mat/fun/inverse_spd.hpp>
#include <stan/math/fwd/mat/fun/inverse.hpp>
#include <stan/math/fwd/mat/fun/typedefs.hpp>
#include <stan/math/fwd/scal/fun/abs.hpp>
#include <stan/math/fwd/scal/fun/fabs.hpp>
#include <gtest/gtest.h>
#include <stan/math/fwd/core.hpp>


class AgradFwdMatrixInverseSPD : public testing::Test {
  void SetUp() {
  }
};



TEST_F(AgradFwdMatrixInverseSPD, exception_fd) {
  using stan::math::inverse_spd;

  // non-square
  stan::math::matrix_fd m1(2,3);
  m1 << 1, 2, 3, 4, 5, 6;
  EXPECT_THROW(inverse_spd(m1),std::invalid_argument);

  
  // non-symmetric
  stan::math::matrix_fd m2(3,3);
  m2 << 1, 2, 3, 4, 5, 6, 7, 8, 9;  
  EXPECT_THROW(inverse_spd(m2),std::domain_error);

  // not positive definite
  stan::math::matrix_fd m3(3,3);
  m3 << 1, 2, 3,
        2, 4, 5,
        3, 5, 6;
  EXPECT_THROW(inverse_spd(m3),std::domain_error);
}

TEST_F(AgradFwdMatrixInverseSPD, exception_ffd) {
  using stan::math::inverse_spd;

  // non-square
  stan::math::matrix_ffd m1(2,3);
  m1 << 1, 2, 3, 4, 5, 6;
  EXPECT_THROW(inverse_spd(m1),std::invalid_argument);

  
  // non-symmetric
  stan::math::matrix_ffd m2(3,3);
  m2 << 1, 2, 3, 4, 5, 6, 7, 8, 9;  
  EXPECT_THROW(inverse_spd(m2),std::domain_error);

  // not positive definite
  stan::math::matrix_ffd m3(3,3);
  m3 << 1, 2, 3,
        2, 4, 5,
        3, 5, 6;
  EXPECT_THROW(inverse_spd(m3),std::domain_error);
}
TEST_F(AgradFwdMatrixInverseSPD, matrix_fd) {
  using stan::math::inverse_spd;

  stan::math::matrix_fd m1(3,3);
  m1 << 2,-1,0,
    -1,2,-1,
    0,-1,2;
  m1(0,0).d_ = 1.0;
  m1(0,1).d_ = 1.0;
  m1(0,2).d_ = 1.0;
  m1(1,0).d_ = 1.0;
  m1(1,1).d_ = 1.0;
  m1(1,2).d_ = 1.0;
  m1(2,0).d_ = 1.0;
  m1(2,1).d_ = 1.0;
  m1(2,2).d_ = 1.0;

  stan::math::matrix_fd m2 = stan::math::inverse(m1);
  stan::math::matrix_fd m3 = inverse_spd(m1);
  
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      EXPECT_FLOAT_EQ(m2(i,j).val_, m3(i,j).val_);
      EXPECT_FLOAT_EQ(m2(i,j).d_, m3(i,j).d_);
    }
  }
}

TEST_F(AgradFwdMatrixInverseSPD, matrix_ffd) {
  using stan::math::inverse_spd;

  stan::math::matrix_ffd m1(3,3);
  m1 << 2,-1,0,
    -1,2,-1,
    0,-1,2;
  m1(0,0).d_ = 1.0;
  m1(0,1).d_ = 1.0;
  m1(0,2).d_ = 1.0;
  m1(1,0).d_ = 1.0;
  m1(1,1).d_ = 1.0;
  m1(1,2).d_ = 1.0;
  m1(2,0).d_ = 1.0;
  m1(2,1).d_ = 1.0;
  m1(2,2).d_ = 1.0;

  stan::math::matrix_ffd m2 = stan::math::inverse(m1);
  stan::math::matrix_ffd m3 = inverse_spd(m1);
  
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      EXPECT_FLOAT_EQ(m2(i,j).val_.val_, m3(i,j).val_.val_);
      EXPECT_FLOAT_EQ(m2(i,j).d_.val_, m3(i,j).d_.val_);
    }
  }
}
