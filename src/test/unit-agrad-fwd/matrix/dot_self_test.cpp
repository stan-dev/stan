#include <gtest/gtest.h>
#include <stan/agrad/fwd/matrix/dot_self.hpp>
#include <stan/math/matrix/dot_self.hpp>
#include <stan/agrad/fwd/fvar.hpp>
#include <stan/agrad/rev.hpp>
#include <test/unit/agrad/util.hpp>

using stan::agrad::fvar;
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
TEST(AgradFwdMatrixDotSelf, vec_fv_1stDeriv) {
  using stan::math::dot_self;
  using stan::agrad::var;

  fvar<var> a(2.0,1.0);
  fvar<var> b(3.0,1.0);
  fvar<var> c(4.0,1.0);

  Eigen::Matrix<fvar<var>,Eigen::Dynamic,1> v1(1);
  v1 << a;
  EXPECT_NEAR(4.0,dot_self(v1).val_.val(),1E-12);
  EXPECT_NEAR(4.0,dot_self(v1).d_.val(),1E-12);
  Eigen::Matrix<fvar<var>,Eigen::Dynamic,1> v2(2);
  v2 << a,b;
  EXPECT_NEAR(13.0,dot_self(v2).val_.val(),1E-12);
  EXPECT_NEAR(10.0,dot_self(v2).d_.val(),1E-12);
  Eigen::Matrix<fvar<var>,Eigen::Dynamic,1> v3(3);
  v3 << a,b,c;
  EXPECT_NEAR(29.0,dot_self(v3).val_.val(),1E-12);  
  EXPECT_NEAR(18.0,dot_self(v3).d_.val(),1E-12);  

  AVEC z = createAVEC(a.val(),b.val(),c.val());
  VEC h;
  dot_self(v3).val_.grad(z,h);
  EXPECT_FLOAT_EQ(4.0,h[0]);
  EXPECT_FLOAT_EQ(6.0,h[1]);
  EXPECT_FLOAT_EQ(8.0,h[2]);
}
TEST(AgradFwdMatrixDotSelf, vec_fv_2ndDeriv) {
  using stan::math::dot_self;
  using stan::agrad::var;

  fvar<var> a(2.0,1.0);
  fvar<var> b(3.0,1.0);
  fvar<var> c(4.0,1.0);

  Eigen::Matrix<fvar<var>,Eigen::Dynamic,1> v3(3);
  v3 << a,b,c;

  AVEC z = createAVEC(a.val(),b.val(),c.val());
  VEC h;
  dot_self(v3).d_.grad(z,h);
  EXPECT_FLOAT_EQ(2.0,h[0]);
  EXPECT_FLOAT_EQ(2.0,h[1]);
  EXPECT_FLOAT_EQ(2.0,h[2]);
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
TEST(AgradFwdMatrixDotSelf, vec_ffv_1stDeriv) {
  using stan::math::dot_self;
  using stan::agrad::var;

  fvar<fvar<var> > a(2.0,1.0);
  fvar<fvar<var> > b(3.0,1.0);
  fvar<fvar<var> > c(4.0,1.0);

  Eigen::Matrix<fvar<fvar<var> >,Eigen::Dynamic,1> v1(1);
  v1 << a;
  EXPECT_NEAR(4.0,dot_self(v1).val_.val().val(),1E-12);
  EXPECT_NEAR(4.0,dot_self(v1).d_.val().val(),1E-12);
  Eigen::Matrix<fvar<fvar<var> >,Eigen::Dynamic,1> v2(2);
  v2 << a,b;
  EXPECT_NEAR(13.0,dot_self(v2).val_.val().val(),1E-12);
  EXPECT_NEAR(10.0,dot_self(v2).d_.val().val(),1E-12);
  Eigen::Matrix<fvar<fvar<var> >,Eigen::Dynamic,1> v3(3);
  v3 << a,b,c;
  EXPECT_NEAR(29.0,dot_self(v3).val_.val().val(),1E-12);  
  EXPECT_NEAR(18.0,dot_self(v3).d_.val().val(),1E-12);  

  AVEC z = createAVEC(a.val().val(),b.val().val(),c.val().val());
  VEC h;
  dot_self(v3).val_.val().grad(z,h);
  EXPECT_FLOAT_EQ(4.0,h[0]);
  EXPECT_FLOAT_EQ(6.0,h[1]);
  EXPECT_FLOAT_EQ(8.0,h[2]);
}
TEST(AgradFwdMatrixDotSelf, vec_ffv_2ndDeriv_1) {
  using stan::math::dot_self;
  using stan::agrad::var;

  fvar<fvar<var> > a(2.0,1.0);
  fvar<fvar<var> > b(3.0,1.0);
  fvar<fvar<var> > c(4.0,1.0);

  Eigen::Matrix<fvar<fvar<var> >,Eigen::Dynamic,1> v3(3);
  v3 << a,b,c;

  AVEC z = createAVEC(a.val().val(),b.val().val(),c.val().val());
  VEC h;
  dot_self(v3).val().d_.grad(z,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
}

TEST(AgradFwdMatrixDotSelf, vec_ffv_2ndDeriv_2) {
  using stan::math::dot_self;
  using stan::agrad::var;

  fvar<fvar<var> > a(2.0,1.0);
  fvar<fvar<var> > b(3.0,1.0);
  fvar<fvar<var> > c(4.0,1.0);

  Eigen::Matrix<fvar<fvar<var> >,Eigen::Dynamic,1> v3(3);
  v3 << a,b,c;

  AVEC z = createAVEC(a.val().val(),b.val().val(),c.val().val());
  VEC h;
  dot_self(v3).d_.val().grad(z,h);
  EXPECT_FLOAT_EQ(2.0,h[0]);
  EXPECT_FLOAT_EQ(2.0,h[1]);
  EXPECT_FLOAT_EQ(2.0,h[2]);
}
TEST(AgradFwdMatrixDotSelf, vec_ffv_3rdDeriv) {
  using stan::math::dot_self;
  using stan::agrad::var;

  fvar<fvar<var> > a(2.0,1.0);
  fvar<fvar<var> > b(3.0,1.0);
  fvar<fvar<var> > c(4.0,1.0);
  a.val_.d_ = 1.0;
  b.val_.d_ = 1.0;
  c.val_.d_ = 1.0;

  Eigen::Matrix<fvar<fvar<var> >,Eigen::Dynamic,1> v3(3);
  v3 << a,b,c;

  AVEC z = createAVEC(a.val().val(),b.val().val(),c.val().val());
  VEC h;
  dot_self(v3).d_.d_.grad(z,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
}

