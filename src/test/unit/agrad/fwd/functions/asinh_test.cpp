#include <gtest/gtest.h>
#include <stan/agrad/fwd.hpp>
#include <boost/math/special_functions/asinh.hpp>
#include <stan/agrad/rev.hpp>
#include <test/unit/agrad/util.hpp>
#include <test/unit/agrad/fwd/nan_util.hpp>

class AgradFwdAsinh : public testing::Test {
  void SetUp() {
    stan::agrad::recover_memory();
  }
};

TEST_F(AgradFwdAsinh,Fvar) {
  using stan::agrad::fvar;
  using boost::math::asinh;
  using std::sqrt;

  fvar<double> x(0.5,1.0);

  fvar<double> a = asinh(x);
  EXPECT_FLOAT_EQ(asinh(0.5), a.val_);
  EXPECT_FLOAT_EQ(1 / sqrt(1 + (0.5) * (0.5)), a.d_);

  fvar<double> y(-1.2,1.0);

  fvar<double> b = asinh(y);
  EXPECT_FLOAT_EQ(asinh(-1.2), b.val_);
  EXPECT_FLOAT_EQ(1 / sqrt(1 + (-1.2) * (-1.2)), b.d_);

  fvar<double> c = asinh(-x);
  EXPECT_FLOAT_EQ(asinh(-0.5), c.val_);
  EXPECT_FLOAT_EQ(-1 / sqrt(1 + (-0.5) * (-0.5)), c.d_);
}

TEST_F(AgradFwdAsinh,FvarVar_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using boost::math::asinh;

  fvar<var> x(1.5,1.3);
  fvar<var> a = asinh(x);

  EXPECT_FLOAT_EQ(asinh(1.5), a.val_.val());
  EXPECT_FLOAT_EQ(1.3 / sqrt(1.0 + 1.5 * 1.5), a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(1.0 / sqrt(1.0 + 1.5 * 1.5), g[0]);
}

TEST_F(AgradFwdAsinh,FvarVar_2ndDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using boost::math::asinh;

  fvar<var> x(1.5,1.3);
  fvar<var> a = asinh(x);

  EXPECT_FLOAT_EQ(asinh(1.5), a.val_.val());
  EXPECT_FLOAT_EQ(1.3 / sqrt(1.0 + 1.5 * 1.5), a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(1.3 * -0.25601548, g[0]);
}

TEST_F(AgradFwdAsinh,FvarFvarDouble) {
  using stan::agrad::fvar;
  using boost::math::asinh;

  fvar<fvar<double> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 2.0;

  fvar<fvar<double> > a = asinh(x);

  EXPECT_FLOAT_EQ(asinh(1.5), a.val_.val_);
  EXPECT_FLOAT_EQ(2.0 / sqrt(1.0 + 1.5 * 1.5), a.val_.d_);
  EXPECT_FLOAT_EQ(0, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);

  fvar<fvar<double> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 2.0;

  a = asinh(y);
  EXPECT_FLOAT_EQ(asinh(1.5), a.val_.val_);
  EXPECT_FLOAT_EQ(0, a.val_.d_);
  EXPECT_FLOAT_EQ(2.0 / sqrt(1.0 + 1.5 * 1.5), a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);
}

TEST_F(AgradFwdAsinh,FvarFvarVar_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using boost::math::asinh;

  stan::agrad::recover_memory();

  fvar<fvar<var> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 2.0;

  fvar<fvar<var> > a = asinh(x);

  EXPECT_FLOAT_EQ(asinh(1.5), a.val_.val_.val());
  EXPECT_FLOAT_EQ(2.0 / sqrt(1.0 + 1.5 * 1.5), a.val_.d_.val());
  EXPECT_FLOAT_EQ(0, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.val_.grad(p,g);
  stan::agrad::recover_memory();
  EXPECT_FLOAT_EQ(1.0 / sqrt(1.0 + 1.5 * 1.5), g[0]);

  fvar<fvar<var> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 2.0;

  fvar<fvar<var> > b = asinh(y);
  EXPECT_FLOAT_EQ(asinh(1.5), a.val_.val_.val());
  EXPECT_FLOAT_EQ(0, a.val_.d_.val());
  EXPECT_FLOAT_EQ(2.0 / sqrt(1.0 + 1.5 * 1.5), a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC q = createAVEC(y.val_.val_);
  VEC r;
  b.val_.val_.grad(q,r);
  stan::agrad::recover_memory();
  EXPECT_FLOAT_EQ(1.0 / sqrt(1.0 + 1.5 * 1.5), r[0]);
}

TEST_F(AgradFwdAsinh,FvarFvarVar_2ndDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using boost::math::asinh;

  fvar<fvar<var> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 2.0;

  fvar<fvar<var> > a = asinh(x);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(2.0 * -0.25601548, g[0]);

  fvar<fvar<var> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 2.0;

  fvar<fvar<var> > b = asinh(y);

  AVEC q = createAVEC(y.val_.val_);
  VEC r;
  b.d_.val_.grad(q,r);
  EXPECT_FLOAT_EQ(2.0 * -0.25601548, r[0]);
}
TEST_F(AgradFwdAsinh,FvarFvarVar_3rdDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using boost::math::asinh;

  fvar<fvar<var> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 1.0;
  x.d_.val_ = 1.0;

  fvar<fvar<var> > a = asinh(x);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.d_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(0.183805982181141, g[0]);
}
struct asinh_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return asinh(arg1);
  }
};

TEST_F(AgradFwdAsinh,asinh_NaN) {
  asinh_fun asinh_;
  test_nan(asinh_,false);
}
