#include <stan/agrad/fwd/matrix/dot_self.hpp>
#include <gtest/gtest.h>
#include <stan/agrad/fwd/fvar.hpp>

using stan::agrad::fvar;
TEST(AgradFwdMatrix, dot_self_vec) {
  using stan::math::dot_self;

  Eigen::Matrix<fvar<double>,Eigen::Dynamic,1> v1(1);
  v1 << 2.0;
  v1(0).d_ = 1.0;
  EXPECT_NEAR(4.0,dot_self(v1).val_,1E-12);
  EXPECT_NEAR(4.0,dot_self(v1).d_,1E-12);
  Eigen::Matrix<fvar<double>,Eigen::Dynamic,1> v2(2);
  v2 << 2.0, 3.0;
  v2(0).d_ = 1.0;
  v2(1).d_ = 1.0;
  EXPECT_NEAR(13.0,dot_self(v2).val_,1E-12);
  EXPECT_NEAR(10.0,dot_self(v2).d_,1E-12);
  Eigen::Matrix<fvar<double>,Eigen::Dynamic,1> v3(3);
  v3 << 2.0, 3.0, 4.0;
  v3(0).d_ = 1.0;
  v3(1).d_ = 1.0;
  v3(2).d_ = 1.0;
  EXPECT_NEAR(29.0,dot_self(v3).val_,1E-12);  
  EXPECT_NEAR(18.0,dot_self(v3).d_,1E-12);  
}
