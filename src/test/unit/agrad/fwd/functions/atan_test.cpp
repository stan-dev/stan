#include <gtest/gtest.h>
#include <stan/agrad/fwd.hpp>
#include <stan/agrad/rev.hpp>
#include <test/unit/agrad/util.hpp>
#include <test/unit/agrad/fwd/nan_util.hpp>

class AgradFwdAtan : public testing::Test {
  void SetUp() {
    stan::agrad::recover_memory();
  }
};


TEST_F(AgradFwdAtan,Fvar) {
  using stan::agrad::fvar;
  using std::atan;

  fvar<double> x(0.5,1.0);
  
  fvar<double> a = atan(x);
  EXPECT_FLOAT_EQ(atan(0.5), a.val_);
  EXPECT_FLOAT_EQ(1 / (1 + 0.5 * 0.5), a.d_);

  fvar<double> b = 2 * atan(x) + 4;
  EXPECT_FLOAT_EQ(2 * atan(0.5) + 4, b.val_);
  EXPECT_FLOAT_EQ(2 / (1 + 0.5 * 0.5), b.d_);

  fvar<double> c = -atan(x) + 5;
  EXPECT_FLOAT_EQ(-atan(0.5) + 5, c.val_);
  EXPECT_FLOAT_EQ(-1 / (1 + 0.5 * 0.5), c.d_);

  fvar<double> d = -3 * atan(x) + 5 * x;
  EXPECT_FLOAT_EQ(-3 * atan(0.5) + 5 * 0.5, d.val_);
  EXPECT_FLOAT_EQ(-3 / (1 + 0.5 * 0.5) + 5, d.d_);
}

TEST_F(AgradFwdAtan,FvarVar_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using std::atan;

  fvar<var> x(1.5,1.3);
  fvar<var> a = atan(x);

  EXPECT_FLOAT_EQ(atan(1.5), a.val_.val());
  EXPECT_FLOAT_EQ(1.3 / (1.0 + 1.5 * 1.5), a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(1.0 / (1.0 + 1.5 * 1.5), g[0]);
}

TEST_F(AgradFwdAtan,FvarVar_2ndDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using std::atan;

  fvar<var> x(1.5,1.3);
  fvar<var> a = atan(x);

  EXPECT_FLOAT_EQ(atan(1.5), a.val_.val());
  EXPECT_FLOAT_EQ(1.3 / (1.0 + 1.5 * 1.5), a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(1.3 * -0.28402368, g[0]);
}

TEST_F(AgradFwdAtan,FvarFvarDouble) {
  using stan::agrad::fvar;
  using std::atan;

  fvar<fvar<double> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 2.0;

  fvar<fvar<double> > a = atan(x);

  EXPECT_FLOAT_EQ(atan(1.5), a.val_.val_);
  EXPECT_FLOAT_EQ(2.0 / (1.0 + 1.5 * 1.5), a.val_.d_);
  EXPECT_FLOAT_EQ(0, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);

  fvar<fvar<double> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 2.0;

  a = atan(y);
  EXPECT_FLOAT_EQ(atan(1.5), a.val_.val_);
  EXPECT_FLOAT_EQ(0, a.val_.d_);
  EXPECT_FLOAT_EQ(2.0 / (1.0 + 1.5 * 1.5), a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);
}

TEST_F(AgradFwdAtan,FvarFvarVar_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using std::atan;

  fvar<fvar<var> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 2.0;

  fvar<fvar<var> > a = atan(x);

  EXPECT_FLOAT_EQ(atan(1.5), a.val_.val_.val());
  EXPECT_FLOAT_EQ(2.0 / (1.0 + 1.5 * 1.5), a.val_.d_.val());
  EXPECT_FLOAT_EQ(0, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.val_.grad(p,g);

  stan::agrad::recover_memory();
  EXPECT_FLOAT_EQ(1.0 / (1.0 + 1.5 * 1.5), g[0]);

  fvar<fvar<var> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 2.0;

  fvar<fvar<var> > b = atan(y);
  EXPECT_FLOAT_EQ(atan(1.5), a.val_.val_.val());
  EXPECT_FLOAT_EQ(0, a.val_.d_.val());
  EXPECT_FLOAT_EQ(2.0 / (1.0 + 1.5 * 1.5), a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC q = createAVEC(y.val_.val_);
  VEC r;
  b.val_.val_.grad(q,r);
  EXPECT_FLOAT_EQ(1.0 / (1.0 + 1.5 * 1.5), r[0]);
}

TEST_F(AgradFwdAtan,FvarFvarVar_2ndDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using std::atan;

  fvar<fvar<var> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 2.0;

  fvar<fvar<var> > a = atan(x);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(2.0 * -0.28402368, g[0]);

  fvar<fvar<var> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 2.0;

  fvar<fvar<var> > b = atan(y);

  AVEC q = createAVEC(y.val_.val_);
  VEC r;
  b.d_.val_.grad(q,r);
  EXPECT_FLOAT_EQ(2.0 * -0.28402368, r[0]);
}
TEST_F(AgradFwdAtan,FvarFvarVar_3rdDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using std::atan;

  fvar<fvar<var> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 1.0;
  x.d_.val_ = 1.0;

  fvar<fvar<var> > a = atan(x);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.d_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(0.335002275830678, g[0]);
}

struct atan_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return atan(arg1);
  }
};

TEST_F(AgradFwdAtan,atan_NaN) {
  atan_fun atan_;
  test_nan(atan_,false);
}
