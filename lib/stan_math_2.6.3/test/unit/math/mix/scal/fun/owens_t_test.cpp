#include <gtest/gtest.h>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <test/unit/math/mix/scal/fun/nan_util.hpp>
#include <stan/math/fwd/scal/fun/owens_t.hpp>
#include <stan/math/rev/scal/fun/owens_t.hpp>
#include <stan/math/fwd/scal/fun/erf.hpp>
#include <stan/math/rev/scal/fun/erf.hpp>
#include <stan/math/fwd/scal/fun/exp.hpp>
#include <stan/math/rev/scal/fun/exp.hpp>


TEST(AgradFwdOwensT,FvarVar_FvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::owens_t;
  using boost::math::owens_t;

  fvar<var> h(1.0,1.0);
  fvar<var> a(2.0,1.0);
  fvar<var> f = owens_t(h,a);
  EXPECT_FLOAT_EQ(owens_t(1.0, 2.0), f.val_.val());
  EXPECT_FLOAT_EQ(0.0026128467 - 0.1154804963, f.d_.val());

  AVEC x = createAVEC(h.val_,a.val_);
  VEC grad_f;
  f.val_.grad(x,grad_f);
  EXPECT_FLOAT_EQ(0.0026128467,grad_f[1]);
  EXPECT_FLOAT_EQ(-0.1154804963,grad_f[0]);
}
TEST(AgradFwdOwensT,FvarVar_Double_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::owens_t;
  using boost::math::owens_t;

  fvar<var> h(1.0,1.0);
  double a(2.0);
  fvar<var> f = owens_t(h,a);
  EXPECT_FLOAT_EQ(owens_t(1.0, 2.0), f.val_.val());
  EXPECT_FLOAT_EQ(-0.1154804963, f.d_.val());

  AVEC x = createAVEC(h.val_);
  VEC grad_f;
  f.val_.grad(x,grad_f);
  EXPECT_FLOAT_EQ(-0.1154804963,grad_f[0]);
}
TEST(AgradFwdOwensT,Double_FvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::owens_t;
  using boost::math::owens_t;

  double h(1.0);
  fvar<var> a(2.0,1.0);
  fvar<var> f = owens_t(h,a);
  EXPECT_FLOAT_EQ(owens_t(1.0, 2.0), f.val_.val());
  EXPECT_FLOAT_EQ(0.0026128467, f.d_.val());

  AVEC x = createAVEC(a.val_);
  VEC grad_f;
  f.val_.grad(x,grad_f);
  EXPECT_FLOAT_EQ(0.0026128467,grad_f[0]);
}
TEST(AgradFwdOwensT,FvarVar_FvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::owens_t;
  using boost::math::owens_t;

  fvar<var> h(1.0,1.0);
  fvar<var> a(2.0,1.0);
  fvar<var> f = owens_t(h,a);

  AVEC x = createAVEC(h.val_,a.val_);
  VEC grad_f;
  f.d_.grad(x,grad_f);
  EXPECT_FLOAT_EQ(-0.020380205,grad_f[1]);
  EXPECT_FLOAT_EQ(0.076287799,grad_f[0]);
}
TEST(AgradFwdOwensT,FvarVar_Double_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::owens_t;
  using boost::math::owens_t;

  fvar<var> h(1.0,1.0);
  double a(2.0);
  fvar<var> f = owens_t(h,a);

  AVEC x = createAVEC(h.val_);
  VEC grad_f;
  f.d_.grad(x,grad_f);
  EXPECT_FLOAT_EQ(0.089352027,grad_f[0]);
}
TEST(AgradFwdOwensT,Double_FvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::owens_t;
  using boost::math::owens_t;

  double h(1.0);
  fvar<var> a(2.0,1.0);
  fvar<var> f = owens_t(h,a);

  AVEC x = createAVEC(a.val_);
  VEC grad_f;
  f.d_.grad(x,grad_f);
  EXPECT_FLOAT_EQ(-0.0073159705,grad_f[0]);
}

TEST(AgradFwdOwensT,FvarFvarVar_FvarFvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::owens_t;
  using boost::math::owens_t;

  fvar<fvar<var> > h,a;
  h.val_.val_ = 1.0;
  h.val_.d_ = 1.0;
  a.val_.val_ = 2.0;
  a.d_.val_ = 1.0;

  fvar<fvar<var> > f = owens_t(h,a);

  EXPECT_FLOAT_EQ(owens_t(1.0, 2.0), f.val_.val_.val());
  EXPECT_FLOAT_EQ(-0.1154804963, f.val_.d_.val());
  EXPECT_FLOAT_EQ(0.0026128467, f.d_.val_.val());
  EXPECT_FLOAT_EQ(-0.013064234,f.d_.d_.val());

  AVEC p = createAVEC(h.val_.val_,a.val_.val_);
  VEC g;
  f.val_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(-0.1154804963, g[0]);
  EXPECT_FLOAT_EQ(0.0026128467, g[1]);
}
TEST(AgradFwdOwensT,FvarFvarVar_Double_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::owens_t;
  using boost::math::owens_t;

  fvar<fvar<var> > h;
  h.val_.val_ = 1.0;
  h.val_.d_ = 1.0;
  double a(2.0);

  fvar<fvar<var> > f = owens_t(h,a);

  EXPECT_FLOAT_EQ(owens_t(1.0, 2.0), f.val_.val_.val());
  EXPECT_FLOAT_EQ(-0.1154804963, f.val_.d_.val());
  EXPECT_FLOAT_EQ(0, f.d_.val_.val());
  EXPECT_FLOAT_EQ(0,f.d_.d_.val());

  AVEC p = createAVEC(h.val_.val_);
  VEC g;
  f.val_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(-0.1154804963, g[0]);
}

TEST(AgradFwdOwensT,Double_FvarFvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::owens_t;
  using boost::math::owens_t;

  double h(1.0);
  fvar<fvar<var> > a;
  a.val_.val_ = 2.0;
  a.d_.val_ = 1.0;

  fvar<fvar<var> > f = owens_t(h,a);

  EXPECT_FLOAT_EQ(owens_t(1.0, 2.0), f.val_.val_.val());
  EXPECT_FLOAT_EQ(0, f.val_.d_.val());
  EXPECT_FLOAT_EQ(0.0026128467, f.d_.val_.val());
  EXPECT_FLOAT_EQ(0,f.d_.d_.val());

  AVEC p = createAVEC(a.val_.val_);
  VEC g;
  f.val_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(0.0026128467, g[0]);
}
TEST(AgradFwdOwensT,FvarFvarVar_FvarFvarVar_2ndDeriv_h) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::owens_t;
  using boost::math::owens_t;

  fvar<fvar<var> > h,a;
  h.val_.val_ = 1.0;
  h.val_.d_ = 1.0;
  a.val_.val_ = 2.0;
  a.d_.val_ = 1.0;

  fvar<fvar<var> > f = owens_t(h,a);

  AVEC p = createAVEC(h.val_.val_,a.val_.val_);
  VEC g;
  f.val_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(0.089352027, g[0]);
  EXPECT_FLOAT_EQ(-0.013064234, g[1]);
}
TEST(AgradFwdOwensT,FvarFvarVar_FvarFvarVar_2ndDeriv_a) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::owens_t;
  using boost::math::owens_t;

  fvar<fvar<var> > h,a;
  h.val_.val_ = 1.0;
  h.val_.d_ = 1.0;
  a.val_.val_ = 2.0;
  a.d_.val_ = 1.0;

  fvar<fvar<var> > f = owens_t(h,a);

  AVEC p = createAVEC(h.val_.val_,a.val_.val_);
  VEC g;
  f.d_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(-0.013064234, g[0]);
  EXPECT_FLOAT_EQ(-0.0073159705, g[1]);
}
TEST(AgradFwdOwensT,FvarFvarVar_Double_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::owens_t;
  using boost::math::owens_t;

  fvar<fvar<var> > h;
  h.val_.val_ = 1.0;
  h.val_.d_ = 1.0;
  double a(2.0);

  fvar<fvar<var> > f = owens_t(h,a);

  AVEC p = createAVEC(h.val_.val_);
  VEC g;
  f.val_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(0.089352027, g[0]);
}

TEST(AgradFwdOwensT,Double_FvarFvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::owens_t;
  using boost::math::owens_t;

  double h(1.0);
  fvar<fvar<var> > a;
  a.val_.val_ = 2.0;
  a.d_.val_ = 1.0;

  fvar<fvar<var> > f = owens_t(h,a);

  AVEC p = createAVEC(a.val_.val_);
  VEC g;
  f.d_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(-0.0073159705, g[0]);
}
TEST(AgradFwdOwensT,FvarFvarVar_FvarFvarVar_3rdDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::owens_t;
  using boost::math::owens_t;

  fvar<fvar<var> > h,a;
  h.val_.val_ = 1.0;
  h.val_.d_ = 1.0;
  a.val_.val_ = 2.0;
  a.d_.val_ = 1.0;

  fvar<fvar<var> > f = owens_t(h,a);

  AVEC p = createAVEC(h.val_.val_,a.val_.val_);
  VEC g;
  f.d_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(0.052256934, g[0]);
  EXPECT_FLOAT_EQ(0.026128467, g[1]);
}
TEST(AgradFwdOwensT,FvarFvarVar_Double_3rdDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::owens_t;
  using boost::math::owens_t;

  fvar<fvar<var> > h;
  h.val_.val_ = 1.0;
  h.val_.d_ = 1.0;
  h.d_.val_ = 1.0;
  double a(2.0);

  fvar<fvar<var> > f = owens_t(h,a);

  AVEC p = createAVEC(h.val_.val_);
  VEC g;
  f.d_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(0.1567708, g[0]);
}

TEST(AgradFwdOwensT,Double_FvarFvarVar_3rdDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::owens_t;
  using boost::math::owens_t;

  double h(1.0);
  fvar<fvar<var> > a;
  a.val_.val_ = 2.0;
  a.d_.val_ = 1.0;
  a.val_.d_ = 1.0;

  fvar<fvar<var> > f = owens_t(h,a);

  AVEC p = createAVEC(a.val_.val_);
  VEC g;
  f.d_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(0.018498953, g[0]);
}

struct owens_t_fun {
  template <typename T0, typename T1>
  inline 
  typename boost::math::tools::promote_args<T0,T1>::type
  operator()(const T0 arg1,
             const T1 arg2) const {
    return owens_t(arg1,arg2);
  }
};

TEST(AgradFwdOwensT, nan) {
  owens_t_fun owens_t_;
  test_nan_mix(owens_t_,3.0,5.0,false);
}
