#include <gtest/gtest.h>
#include <stan/math/fwd/mat/fun/dot_self.hpp>
#include <stan/math/prim/mat/fun/dot_self.hpp>
#include <stan/math/fwd/core.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <stan/math/rev/core.hpp>

using stan::math::fvar;
TEST(AgradMixMatrixDotSelf, vec_fv_1stDeriv) {
  using stan::math::dot_self;
  using stan::math::var;

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
TEST(AgradMixMatrixDotSelf, vec_fv_2ndDeriv) {
  using stan::math::dot_self;
  using stan::math::var;

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
TEST(AgradMixMatrixDotSelf, vec_ffv_1stDeriv) {
  using stan::math::dot_self;
  using stan::math::var;

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
TEST(AgradMixMatrixDotSelf, vec_ffv_2ndDeriv_1) {
  using stan::math::dot_self;
  using stan::math::var;

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

TEST(AgradMixMatrixDotSelf, vec_ffv_2ndDeriv_2) {
  using stan::math::dot_self;
  using stan::math::var;

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
TEST(AgradMixMatrixDotSelf, vec_ffv_3rdDeriv) {
  using stan::math::dot_self;
  using stan::math::var;

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

