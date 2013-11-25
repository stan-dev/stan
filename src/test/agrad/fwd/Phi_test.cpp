#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>
#include <stan/math/functions/Phi.hpp>
#include <stan/prob/distributions/univariate/continuous/normal.hpp>
#include <stan/agrad/var.hpp>
#include <test/agrad/util.hpp>

TEST(AgradFwdPhi,Fvar) {
  using stan::agrad::fvar;
  using stan::math::Phi;
  fvar<double> x = 1.0;
  x.d_ = 1.0;
  
  fvar<double> Phi_x = Phi(x);

  EXPECT_FLOAT_EQ(Phi(1.0), Phi_x.val_);
  EXPECT_FLOAT_EQ(exp(stan::prob::normal_log<false>(1.0,0.0,1.0)),
                  Phi_x.d_);
}
TEST(AgradFwdPhi, FvarVar_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::Phi;

  fvar<var> x(1.0,1.3);
  fvar<var> a = Phi(x);

  EXPECT_FLOAT_EQ(Phi(1.0), a.val_.val());
  EXPECT_FLOAT_EQ(1.3 * exp(stan::prob::normal_log<false>(1.0,0.0,1.0)), 
                  a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(exp(stan::prob::normal_log<false>(1.0,0.0,1.0)), g[0]);
}
TEST(AgradFwdPhi, FvarVar_2ndDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::Phi;

  fvar<var> x(1.0,1.3);
  fvar<var> a = Phi(x);

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(-1.3 * exp(stan::prob::normal_log<false>(1.0,0.0,1.0)), g[0]);
}
TEST(AgradFwdPhi, FvarFvarDouble) {
  using stan::agrad::fvar;
  using stan::math::Phi;

  fvar<fvar<double> > x;
  x.val_.val_ = 1.0;
  x.val_.d_ = 1.0;

  fvar<fvar<double> > a = Phi(x);

  EXPECT_FLOAT_EQ(Phi(1.0), a.val_.val_);
  EXPECT_FLOAT_EQ(exp(stan::prob::normal_log<false>(1.0,0.0,1.0)), a.val_.d_);
  EXPECT_FLOAT_EQ(0, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);

  fvar<fvar<double> > y;
  y.val_.val_ = 1.0;
  y.d_.val_ = 1.0;

  a = Phi(y);
  EXPECT_FLOAT_EQ(Phi(1.0), a.val_.val_);
  EXPECT_FLOAT_EQ(0, a.val_.d_);
  EXPECT_FLOAT_EQ(exp(stan::prob::normal_log<false>(1.0,0.0,1.0)), a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);
}
TEST(AgradFwdPhi, FvarFvarVar_1stDeriv) {
  using stan::agrad::fvar;
  using stan::math::Phi;
  using stan::agrad::var;

  fvar<fvar<var> > x;
  x.val_.val_ = 1.0;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > a = Phi(x);

  EXPECT_FLOAT_EQ(Phi(1.0), a.val_.val_.val());
  EXPECT_FLOAT_EQ(exp(stan::prob::normal_log<false>(1.0,0.0,1.0)), 
                  a.val_.d_.val());
  EXPECT_FLOAT_EQ(0, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(exp(stan::prob::normal_log<false>(1.0,0.0,1.0)), g[0]);

  fvar<fvar<var> > y;
  y.val_.val_ = 1.0;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > b = Phi(y);
  EXPECT_FLOAT_EQ(Phi(1.0), b.val_.val_.val());
  EXPECT_FLOAT_EQ(0, b.val_.d_.val());
  EXPECT_FLOAT_EQ(exp(stan::prob::normal_log<false>(1.0,0.0,1.0)),
                  b.d_.val_.val());
  EXPECT_FLOAT_EQ(0, b.d_.d_.val());

  AVEC q = createAVEC(y.val_.val_);
  VEC r;
  b.val_.val_.grad(q,r);
  EXPECT_FLOAT_EQ(exp(stan::prob::normal_log<false>(1.0,0.0,1.0)), r[0]);
}

TEST(AgradFwdPhi, FvarFvarVar_2ndDeriv) {
  using stan::agrad::fvar;
  using stan::math::Phi;
  using stan::agrad::var;

  fvar<fvar<var> > x;
  x.val_.val_ = 1.0;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > a = Phi(x);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(-exp(stan::prob::normal_log<false>(1.0,0.0,1.0)), g[0]);

  fvar<fvar<var> > y;
  y.val_.val_ = 1.0;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > b = Phi(y);

  AVEC q = createAVEC(y.val_.val_);
  VEC r;
  b.d_.val_.grad(q,r);
  EXPECT_FLOAT_EQ(-exp(stan::prob::normal_log<false>(1.0,0.0,1.0)), r[0]);
}
TEST(AgradFwdPhi, FvarFvarVar_3rdDeriv) {
  using stan::agrad::fvar;
  using stan::math::Phi;
  using stan::agrad::var;

  fvar<fvar<var> > x;
  x.val_.val_ = 1.0;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > a = Phi(x);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.d_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(0, g[0]);
}
