#include <stan/math/rev/core.hpp>
#include <stan/math/prim/mat/fun/inverse_spd.hpp>
#include <stan/math/fwd/mat/fun/inverse.hpp>
#include <stan/math/rev/mat/fun/multiply.hpp>
#include <stan/math/fwd/mat/fun/typedefs.hpp>
#include <stan/math/mix/mat/fun/typedefs.hpp>
#include <stan/math/fwd/scal/fun/abs.hpp>
#include <stan/math/rev/scal/fun/abs.hpp>
#include <stan/math/fwd/scal/fun/fabs.hpp>
#include <stan/math/rev/scal/fun/fabs.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <gtest/gtest.h>
#include <stan/math/fwd/core.hpp>

using stan::math::var;

class AgradMixMatrixInverseSPD : public testing::Test {
  void SetUp() {
    stan::math::recover_memory();
  }
};

TEST_F(AgradMixMatrixInverseSPD, exception_fv) {
  using stan::math::inverse_spd;

  // non-square
  stan::math::matrix_fv m1(2,3);
  m1 << 1, 2, 3, 4, 5, 6;
  EXPECT_THROW(inverse_spd(m1),std::invalid_argument);

  // non-symmetric
  stan::math::matrix_fv m2(3,3);
  m2 << 1, 2, 3, 4, 5, 6, 7, 8, 9;
  EXPECT_THROW(inverse_spd(m2),std::domain_error);

  // not positive definite
  stan::math::matrix_fv m3(3,3);
  m3 << 1, 2, 3,
        2, 4, 5,
        3, 5, 6;
  EXPECT_THROW(inverse_spd(m3),std::domain_error);
}
TEST_F(AgradMixMatrixInverseSPD, exception_ffv) {
  using stan::math::inverse_spd;

  // non-square
  stan::math::matrix_ffv m1(2,3);
  m1 << 1, 2, 3, 4, 5, 6;
  EXPECT_THROW(inverse_spd(m1),std::invalid_argument);

  
  // non-symmetric
  stan::math::matrix_ffv m2(3,3);
  m2 << 1, 2, 3, 4, 5, 6, 7, 8, 9;  
  EXPECT_THROW(inverse_spd(m2),std::domain_error);

  // not positive definite 
  stan::math::matrix_ffv m3(3,3);
  m3 << 1, 2, 3,
        2, 4, 5,
        3, 5, 6;
  EXPECT_THROW(inverse_spd(m3),std::domain_error);
}
TEST_F(AgradMixMatrixInverseSPD, matrix_fv_1st_deriv) {
  using stan::math::inverse_spd;

  stan::math::matrix_fv m1(3,3);
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

  stan::math::matrix_fv m2 = stan::math::inverse(m1);
  stan::math::matrix_fv m3 = inverse_spd(m1);
  
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      EXPECT_FLOAT_EQ(m2(i,j).val_.val(), m3(i,j).val_.val());
      EXPECT_FLOAT_EQ(m2(i,j).d_.val(), m3(i,j).d_.val());
    }
  }

  std::vector<var> z1;
  z1.push_back(m1(0,0).val_);
  z1.push_back(m1(0,1).val_);
  z1.push_back(m1(0,2).val_);
  z1.push_back(m1(1,0).val_);
  z1.push_back(m1(1,1).val_);
  z1.push_back(m1(1,2).val_);
  z1.push_back(m1(2,0).val_);
  z1.push_back(m1(2,1).val_);
  z1.push_back(m1(2,2).val_);

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      VEC h1;
      VEC h2;
      m2(i,j).val_.grad(z1,h1);
      stan::math::recover_memory();
      m3(i,j).val_.grad(z1,h2);
      stan::math::recover_memory();
      for (int k = 0; k < 9; k++)
        EXPECT_FLOAT_EQ(h1[k], h2[k]);
    }
  }
}

TEST_F(AgradMixMatrixInverseSPD, matrix_fv_2nd_deriv) {
  using stan::math::inverse_spd;

  stan::math::matrix_fv m1(3,3);
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

  stan::math::matrix_fv m2 = stan::math::inverse(m1);
  stan::math::matrix_fv m3 = inverse_spd(m1);
  
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      EXPECT_FLOAT_EQ(m2(i,j).val_.val(), m3(i,j).val_.val());
      EXPECT_FLOAT_EQ(m2(i,j).d_.val(), m3(i,j).d_.val());
    }
  }

  std::vector<var> z1;
  z1.push_back(m1(0,0).val_);
  z1.push_back(m1(0,1).val_);
  z1.push_back(m1(0,2).val_);
  z1.push_back(m1(1,0).val_);
  z1.push_back(m1(1,1).val_);
  z1.push_back(m1(1,2).val_);
  z1.push_back(m1(2,0).val_);
  z1.push_back(m1(2,1).val_);
  z1.push_back(m1(2,2).val_);

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      VEC h1;
      VEC h2;
      m2(i,j).d_.grad(z1,h1);
      stan::math::recover_memory();
      m3(i,j).d_.grad(z1,h2);
      stan::math::recover_memory();
      for (int k = 0; k < 9; k++)
        EXPECT_FLOAT_EQ(h1[k], h2[k]);
    }
  }
}

TEST_F(AgradMixMatrixInverseSPD, matrix_ffv_1st_deriv) {
  using stan::math::inverse_spd;

  stan::math::matrix_ffv m1(3,3);
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

  stan::math::matrix_ffv m2 = stan::math::inverse(m1);
  stan::math::matrix_ffv m3 = inverse_spd(m1);
  
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      EXPECT_FLOAT_EQ(m2(i,j).val_.val_.val(), m3(i,j).val_.val_.val());
      EXPECT_FLOAT_EQ(m2(i,j).d_.val_.val(), m3(i,j).d_.val_.val());
    }
  }

  std::vector<var> z1;
  z1.push_back(m1(0,0).val_.val_);
  z1.push_back(m1(0,1).val_.val_);
  z1.push_back(m1(0,2).val_.val_);
  z1.push_back(m1(1,0).val_.val_);
  z1.push_back(m1(1,1).val_.val_);
  z1.push_back(m1(1,2).val_.val_);
  z1.push_back(m1(2,0).val_.val_);
  z1.push_back(m1(2,1).val_.val_);
  z1.push_back(m1(2,2).val_.val_);

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      VEC h1;
      VEC h2;
      m2(i,j).val_.val_.grad(z1,h1);
      stan::math::recover_memory();
      m3(i,j).val_.val_.grad(z1,h2);
      stan::math::recover_memory();
      for (int k = 0; k < 9; k++)
        EXPECT_FLOAT_EQ(h1[k], h2[k]);
    }
  }
}

TEST_F(AgradMixMatrixInverseSPD, matrix_ffv_2nd_deriv) {
  using stan::math::inverse_spd;

  stan::math::matrix_ffv m1(3,3);
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

  stan::math::matrix_ffv m2 = stan::math::inverse(m1);
  stan::math::matrix_ffv m3 = inverse_spd(m1);
  
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      EXPECT_FLOAT_EQ(m2(i,j).val_.val_.val(), m3(i,j).val_.val_.val());
      EXPECT_FLOAT_EQ(m2(i,j).d_.val_.val(), m3(i,j).d_.val_.val());
    }
  }

  std::vector<var> z1;
  z1.push_back(m1(0,0).val_.val_);
  z1.push_back(m1(0,1).val_.val_);
  z1.push_back(m1(0,2).val_.val_);
  z1.push_back(m1(1,0).val_.val_);
  z1.push_back(m1(1,1).val_.val_);
  z1.push_back(m1(1,2).val_.val_);
  z1.push_back(m1(2,0).val_.val_);
  z1.push_back(m1(2,1).val_.val_);
  z1.push_back(m1(2,2).val_.val_);

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      VEC h1;
      VEC h2;
      m2(i,j).d_.val_.grad(z1,h1);
      stan::math::recover_memory();
      m3(i,j).d_.val_.grad(z1,h2);
      stan::math::recover_memory();
      for (int k = 0; k < 9; k++)
        EXPECT_FLOAT_EQ(h1[k], h2[k]);
    }
  }
}

TEST_F(AgradMixMatrixInverseSPD, matrix_ffv_3rd_deriv) {
  using stan::math::inverse_spd;

  stan::math::matrix_ffv m1(3,3);
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
  m1(0,0).val_.d_ = 1.0;
  m1(0,1).val_.d_ = 1.0;
  m1(0,2).val_.d_ = 1.0;
  m1(1,0).val_.d_ = 1.0;
  m1(1,1).val_.d_ = 1.0;
  m1(1,2).val_.d_ = 1.0;
  m1(2,0).val_.d_ = 1.0;
  m1(2,1).val_.d_ = 1.0;
  m1(2,2).val_.d_ = 1.0;

  stan::math::matrix_ffv m2 = stan::math::inverse(m1);
  stan::math::matrix_ffv m3 = inverse_spd(m1);
  
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      EXPECT_FLOAT_EQ(m2(i,j).val_.val_.val(), m3(i,j).val_.val_.val());
      EXPECT_FLOAT_EQ(m2(i,j).d_.val_.val(), m3(i,j).d_.val_.val());
    }
  }

  std::vector<var> z1;
  z1.push_back(m1(0,0).val_.val_);
  z1.push_back(m1(0,1).val_.val_);
  z1.push_back(m1(0,2).val_.val_);
  z1.push_back(m1(1,0).val_.val_);
  z1.push_back(m1(1,1).val_.val_);
  z1.push_back(m1(1,2).val_.val_);
  z1.push_back(m1(2,0).val_.val_);
  z1.push_back(m1(2,1).val_.val_);
  z1.push_back(m1(2,2).val_.val_);

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      VEC h1;
      VEC h2;
      m2(i,j).d_.d_.grad(z1,h1);
      stan::math::recover_memory();
      m3(i,j).d_.d_.grad(z1,h2);
      stan::math::recover_memory();
      for (int k = 0; k < 9; k++)
        EXPECT_FLOAT_EQ(h1[k], h2[k]);
    }
  }
}
