#include <gtest/gtest.h>
#include <stan/math/fwd/mat/fun/dot_self.hpp>
#include <stan/math/prim/mat/fun/dot_self.hpp>
#include <stan/math/fwd/core.hpp>

using stan::math::fvar;
TEST(AgradFwdMatrixDotSelf, vec_fd) {
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
TEST(AgradFwdMatrixDotSelf, vec_ffd) {
  using stan::math::dot_self;

  fvar<fvar<double> > a;
  fvar<fvar<double> > b;
  fvar<fvar<double> > c;
  a.val_.val_ = 2.0;
  a.d_.val_ = 1.0;  
  b.val_.val_ = 3.0;
  b.d_.val_ = 1.0;
  c.val_.val_ = 4.0;
  c.d_.val_ = 1.0;

  Eigen::Matrix<fvar<fvar<double> >,Eigen::Dynamic,1> v1(1);
  v1 << a;
  EXPECT_NEAR(4.0,dot_self(v1).val_.val(),1E-12);
  EXPECT_NEAR(4.0,dot_self(v1).d_.val(),1E-12);
  Eigen::Matrix<fvar<fvar<double> >,Eigen::Dynamic,1> v2(2);
  v2 << a,b;
  EXPECT_NEAR(13.0,dot_self(v2).val_.val(),1E-12);
  EXPECT_NEAR(10.0,dot_self(v2).d_.val(),1E-12);
  Eigen::Matrix<fvar<fvar<double> >,Eigen::Dynamic,1> v3(3);
  v3 << a,b,c;
  EXPECT_NEAR(29.0,dot_self(v3).val_.val(),1E-12);  
  EXPECT_NEAR(18.0,dot_self(v3).d_.val(),1E-12);  
}
