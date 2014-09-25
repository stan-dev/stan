#include <stan/math/matrix/inverse_spd.hpp>
#include <stan/agrad/fwd/matrix/inverse.hpp>
#include <stan/agrad/rev/matrix/multiply.hpp>
#include <stan/agrad/fwd/matrix/typedefs.hpp>
#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/fwd/operators.hpp>
#include <stan/agrad/rev/operators.hpp>
#include <stan/agrad/fwd/functions/abs.hpp>
#include <stan/agrad/rev/functions/abs.hpp>
#include <stan/agrad/fwd/functions/fabs.hpp>
#include <stan/agrad/rev/functions/fabs.hpp>
#include <test/unit/agrad/util.hpp>
#include <gtest/gtest.h>

using stan::agrad::var;

class AgradFwdMatrixInverseSPD : public testing::Test {
  void SetUp() {
    stan::agrad::recover_memory();
  }
};



TEST_F(AgradFwdMatrixInverseSPD, exception_fd) {
  using stan::math::inverse_spd;

  stan::agrad::matrix_fd m1(2,3);
  
  // non-square
  m1 << 1, 2, 3, 4, 5, 6;
  EXPECT_THROW(inverse_spd(m1),std::domain_error);

  stan::agrad::matrix_fd m2(3,3);
  
  // non-symmetric
  m2 << 1, 2, 3, 4, 5, 6, 7, 8, 9;  
  EXPECT_THROW(inverse_spd(m1),std::domain_error);

  // not positive definite
  m2 << 1, 2, 3,
        2, 4, 5,
        3, 5, 6;
  EXPECT_THROW(inverse_spd(m1),std::domain_error);
}

TEST_F(AgradFwdMatrixInverseSPD, exception_ffd) {
  using stan::math::inverse_spd;

  stan::agrad::matrix_ffd m1(2,3);
  
  // non-square
  m1 << 1, 2, 3, 4, 5, 6;
  EXPECT_THROW(inverse_spd(m1),std::domain_error);

  stan::agrad::matrix_ffd m2(3,3);
  
  // non-symmetric
  m2 << 1, 2, 3, 4, 5, 6, 7, 8, 9;  
  EXPECT_THROW(inverse_spd(m1),std::domain_error);

  // not positive definite
  m2 << 1, 2, 3,
        2, 4, 5,
        3, 5, 6;
  EXPECT_THROW(inverse_spd(m1),std::domain_error);
}
TEST_F(AgradFwdMatrixInverseSPD, exception_fv) {
  using stan::math::inverse_spd;

  stan::agrad::matrix_fv m1(2,3);
  
  // non-square
  m1 << 1, 2, 3, 4, 5, 6;
  EXPECT_THROW(inverse_spd(m1),std::domain_error);

  stan::agrad::matrix_fv m2(3,3);
  
  // non-symmetric
  m2 << 1, 2, 3, 4, 5, 6, 7, 8, 9;  
  EXPECT_THROW(inverse_spd(m1),std::domain_error);

  // not positive definite
  m2 << 1, 2, 3,
        2, 4, 5,
        3, 5, 6;
  EXPECT_THROW(inverse_spd(m1),std::domain_error);
}
TEST_F(AgradFwdMatrixInverseSPD, exception_ffv) {
  using stan::math::inverse_spd;

  stan::agrad::matrix_ffv m1(2,3);
  
  // non-square
  m1 << 1, 2, 3, 4, 5, 6;
  EXPECT_THROW(inverse_spd(m1),std::domain_error);

  stan::agrad::matrix_ffv m2(3,3);
  
  // non-symmetric
  m2 << 1, 2, 3, 4, 5, 6, 7, 8, 9;  
  EXPECT_THROW(inverse_spd(m1),std::domain_error);

  // not positive definite
  m2 << 1, 2, 3,
        2, 4, 5,
        3, 5, 6;
  EXPECT_THROW(inverse_spd(m1),std::domain_error);
}

TEST_F(AgradFwdMatrixInverseSPD, matrix_fd) {
  using stan::math::inverse_spd;

  stan::agrad::matrix_fd m1(3,3);
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

  stan::agrad::matrix_fd m2 = stan::agrad::inverse(m1);
  stan::agrad::matrix_fd m3 = inverse_spd(m1);
  
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      EXPECT_FLOAT_EQ(m2(i,j).val_, m3(i,j).val_);
      EXPECT_FLOAT_EQ(m2(i,j).d_, m3(i,j).d_);
    }
  }
}

TEST_F(AgradFwdMatrixInverseSPD, matrix_ffd) {
  using stan::math::inverse_spd;

  stan::agrad::matrix_ffd m1(3,3);
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

  stan::agrad::matrix_ffd m2 = stan::agrad::inverse(m1);
  stan::agrad::matrix_ffd m3 = inverse_spd(m1);
  
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      EXPECT_FLOAT_EQ(m2(i,j).val_.val_, m3(i,j).val_.val_);
      EXPECT_FLOAT_EQ(m2(i,j).d_.val_, m3(i,j).d_.val_);
    }
  }
}

TEST_F(AgradFwdMatrixInverseSPD, matrix_fv_1st_deriv) {
  using stan::math::inverse_spd;

  stan::agrad::matrix_fv m1(3,3);
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

  stan::agrad::matrix_fv m2 = stan::agrad::inverse(m1);
  stan::agrad::matrix_fv m3 = inverse_spd(m1);
  
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
      stan::agrad::recover_memory();
      m3(i,j).val_.grad(z1,h2);
      stan::agrad::recover_memory();
      for (int k = 0; k < 9; k++)
        EXPECT_FLOAT_EQ(h1[k], h2[k]);
    }
  }
}

TEST_F(AgradFwdMatrixInverseSPD, matrix_fv_2nd_deriv) {
  using stan::math::inverse_spd;

  stan::agrad::matrix_fv m1(3,3);
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

  stan::agrad::matrix_fv m2 = stan::agrad::inverse(m1);
  stan::agrad::matrix_fv m3 = inverse_spd(m1);
  
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
      stan::agrad::recover_memory();
      m3(i,j).d_.grad(z1,h2);
      stan::agrad::recover_memory();
      for (int k = 0; k < 9; k++)
        EXPECT_FLOAT_EQ(h1[k], h2[k]);
    }
  }
}

TEST_F(AgradFwdMatrixInverseSPD, matrix_ffv_1st_deriv) {
  using stan::math::inverse_spd;

  stan::agrad::matrix_ffv m1(3,3);
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

  stan::agrad::matrix_ffv m2 = stan::agrad::inverse(m1);
  stan::agrad::matrix_ffv m3 = inverse_spd(m1);
  
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
      stan::agrad::recover_memory();
      m3(i,j).val_.val_.grad(z1,h2);
      stan::agrad::recover_memory();
      for (int k = 0; k < 9; k++)
        EXPECT_FLOAT_EQ(h1[k], h2[k]);
    }
  }
}

TEST_F(AgradFwdMatrixInverseSPD, matrix_ffv_2nd_deriv) {
  using stan::math::inverse_spd;

  stan::agrad::matrix_ffv m1(3,3);
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

  stan::agrad::matrix_ffv m2 = stan::agrad::inverse(m1);
  stan::agrad::matrix_ffv m3 = inverse_spd(m1);
  
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
      stan::agrad::recover_memory();
      m3(i,j).d_.val_.grad(z1,h2);
      stan::agrad::recover_memory();
      for (int k = 0; k < 9; k++)
        EXPECT_FLOAT_EQ(h1[k], h2[k]);
    }
  }
}

TEST_F(AgradFwdMatrixInverseSPD, matrix_ffv_3rd_deriv) {
  using stan::math::inverse_spd;

  stan::agrad::matrix_ffv m1(3,3);
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

  stan::agrad::matrix_ffv m2 = stan::agrad::inverse(m1);
  stan::agrad::matrix_ffv m3 = inverse_spd(m1);
  
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
      stan::agrad::recover_memory();
      m3(i,j).d_.d_.grad(z1,h2);
      stan::agrad::recover_memory();
      for (int k = 0; k < 9; k++)
        EXPECT_FLOAT_EQ(h1[k], h2[k]);
    }
  }
}
