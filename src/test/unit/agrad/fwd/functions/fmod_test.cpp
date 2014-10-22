#include <gtest/gtest.h>
#include <stan/agrad/fwd.hpp>
#include <stan/agrad/rev.hpp>
#include <test/unit/agrad/util.hpp>
#include <test/unit/agrad/fwd/nan_util.hpp>

TEST(AgradFwdFmod,Fvar) {
  using stan::agrad::fvar;
  using std::fmod;
  using std::floor;

  fvar<double> x(2.0,1.0);
  fvar<double> y(3.0,2.0);

  fvar<double> a = fmod(x, y);
  EXPECT_FLOAT_EQ(fmod(2.0, 3.0), a.val_);
  EXPECT_FLOAT_EQ(1.0 * 1.0 + 2.0 * -floor(2.0 / 3.0), a.d_);

  double z = 4.0;
  double w = 3.0;

  fvar<double> e = fmod(x, z);
  EXPECT_FLOAT_EQ(fmod(2.0, 4.0), e.val_);
  EXPECT_FLOAT_EQ(1.0 / 4.0, e.d_);

  fvar<double> f = fmod(w, x);
  EXPECT_FLOAT_EQ(fmod(3.0, 2.0), f.val_);
  EXPECT_FLOAT_EQ(1.0 * -floor(3.0 / 2.0), f.d_);
 }
TEST(AgradFwdFmod,FvarVar_FvarVar_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using std::fmod;

  fvar<var> x(3.0,1.3);
  fvar<var> z(6.0,1.0);
  fvar<var> a = fmod(x,z);

  EXPECT_FLOAT_EQ(fmod(3.0,6.0), a.val_.val());
  EXPECT_FLOAT_EQ(1.3, a.d_.val());

  AVEC y = createAVEC(x.val_,z.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(1,g[0]);
  std::isnan(g[1]);
}
TEST(AgradFwdFmod,FvarVar_Double_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using std::fmod;

  fvar<var> x(3.0,1.3);
  double z(6.0);
  fvar<var> a = fmod(x,z);

  EXPECT_FLOAT_EQ(fmod(3.0,6.0), a.val_.val());
  EXPECT_FLOAT_EQ(13.0 / 60.0, a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(1,g[0]);
}
TEST(AgradFwdFmod,Double_FvarVar_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using std::fmod;

  double x(3.0);
  fvar<var> z(6.0,1.0);
  fvar<var> a = fmod(x,z);

  EXPECT_FLOAT_EQ(fmod(3.0,6.0), a.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.val());

  AVEC y = createAVEC(z.val_);
  VEC g;
  a.val_.grad(y,g);
  std::isnan(g[0]);
}
TEST(AgradFwdFmod,FvarVar_FvarVar_2ndDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using std::fmod;

  fvar<var> x(3.0,1.3);
  fvar<var> z(6.0,1.0);
  fvar<var> a = fmod(x,z);

  EXPECT_FLOAT_EQ(fmod(3.0,6.0), a.val_.val());
  EXPECT_FLOAT_EQ(1.3, a.d_.val());

  AVEC y = createAVEC(x.val_,z.val_);
  VEC g;
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(0,g[0]);
  std::isnan(g[1]);
}
TEST(AgradFwdFmod,FvarVar_Double_2ndDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using std::fmod;

  fvar<var> x(3.0,1.3);
  double z(6.0);
  fvar<var> a = fmod(x,z);

  EXPECT_FLOAT_EQ(fmod(3.0,6.0), a.val_.val());
  EXPECT_FLOAT_EQ(13.0 / 60.0, a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(0,g[0]);
}
TEST(AgradFwdFmod,Double_FvarVar_2ndDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using std::fmod;

  double x(3.0);
  fvar<var> z(6.0,1.0);
  fvar<var> a = fmod(x,z);

  EXPECT_FLOAT_EQ(fmod(3.0,6.0), a.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.val());

  AVEC y = createAVEC(z.val_);
  VEC g;
  a.d_.grad(y,g);
  std::isnan(g[0]);
}
TEST(AgradFwdFmod,FvarFvarDouble) {
  using stan::agrad::fvar;
  using std::fmod;

  fvar<fvar<double> > x;
  x.val_.val_ = 3.0;
  x.val_.d_ = 1.0;

  fvar<fvar<double> > y;
  y.val_.val_ = 6.0;
  y.d_.val_ = 1.0;

  fvar<fvar<double> > a = fmod(x,y);

  EXPECT_FLOAT_EQ(fmod(3.0,6.0), a.val_.val_);
  EXPECT_FLOAT_EQ(1, a.val_.d_);
  EXPECT_FLOAT_EQ(0, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);
}
TEST(AgradFwdFmod,FvarFvarVar_FvarFvarVar_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using std::fmod;

  fvar<fvar<var> > x;
  x.val_.val_ = 3.0;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > y;
  y.val_.val_ = 6.0;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = fmod(x,y);

  EXPECT_FLOAT_EQ(fmod(3.0,6.0), a.val_.val_.val());
  EXPECT_FLOAT_EQ(1, a.val_.d_.val());
  EXPECT_FLOAT_EQ(0, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC q = createAVEC(x.val_.val_,y.val_.val_);
  VEC r;
  a.val_.val_.grad(q,r);
  EXPECT_FLOAT_EQ(1, r[0]);
  EXPECT_FLOAT_EQ(0, r[1]);
}
TEST(AgradFwdFmod,FvarFvarVar_Double_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using std::fmod;

  fvar<fvar<var> > x;
  x.val_.val_ = 3.0;
  x.val_.d_ = 1.0;

  double y(6.0);

  fvar<fvar<var> > a = fmod(x,y);

  EXPECT_FLOAT_EQ(fmod(3.0,6.0), a.val_.val_.val());
  EXPECT_FLOAT_EQ(1.0 / 6.0, a.val_.d_.val());
  EXPECT_FLOAT_EQ(0, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC q = createAVEC(x.val_.val_);
  VEC r;
  a.val_.val_.grad(q,r);
  EXPECT_FLOAT_EQ(1, r[0]);
}
TEST(AgradFwdFmod,Double_FvarFvarVar_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using std::fmod;

  double x(3.0);
  fvar<fvar<var> > y;
  y.val_.val_ = 6.0;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = fmod(x,y);

  EXPECT_FLOAT_EQ(fmod(3.0,6.0), a.val_.val_.val());
  EXPECT_FLOAT_EQ(0, a.val_.d_.val());
  EXPECT_FLOAT_EQ(0, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC q = createAVEC(y.val_.val_);
  VEC r;
  a.val_.val_.grad(q,r);
  EXPECT_FLOAT_EQ(0, r[0]);
}
TEST(AgradFwdFmod,FvarFvarVar_FvarFvarVar_2ndDeriv_x) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using std::fmod;

  fvar<fvar<var> > x;
  x.val_.val_ = 3.0;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > y;
  y.val_.val_ = 6.0;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = fmod(x,y);

  EXPECT_FLOAT_EQ(fmod(3.0,6.0), a.val_.val_.val());
  EXPECT_FLOAT_EQ(1, a.val_.d_.val());
  EXPECT_FLOAT_EQ(0, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC q = createAVEC(x.val_.val_,y.val_.val_);
  VEC r;
  a.val_.d_.grad(q,r);
  EXPECT_FLOAT_EQ(0, r[0]);
  EXPECT_FLOAT_EQ(0, r[1]);
}
TEST(AgradFwdFmod,FvarFvarVar_FvarFvarVar_2ndDeriv_y) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using std::fmod;

  fvar<fvar<var> > x;
  x.val_.val_ = 3.0;
  x.val_.d_ = 1.0;
  fvar<fvar<var> > y;
  y.val_.val_ = 6.0;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = fmod(x,y);

  AVEC q = createAVEC(x.val_.val_,y.val_.val_);
  VEC r;
  a.d_.val_.grad(q,r);
  EXPECT_FLOAT_EQ(0, r[0]);
  EXPECT_FLOAT_EQ(0, r[1]);
}
TEST(AgradFwdFmod,FvarFvarVar_Double_2ndDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using std::fmod;

  fvar<fvar<var> > x;
  x.val_.val_ = 3.0;
  x.val_.d_ = 1.0;

  double y(6.0);

  fvar<fvar<var> > a = fmod(x,y);

  AVEC q = createAVEC(x.val_.val_);
  VEC r;
  a.val_.d_.grad(q,r);
  EXPECT_FLOAT_EQ(0, r[0]);
}
TEST(AgradFwdFmod,Double_FvarFvarVar_2ndDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using std::fmod;

  double x(3.0);
  fvar<fvar<var> > y;
  y.val_.val_ = 6.0;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = fmod(x,y);

  AVEC q = createAVEC(y.val_.val_);
  VEC r;
  a.d_.val_.grad(q,r);
  EXPECT_FLOAT_EQ(0, r[0]);
}
TEST(AgradFwdFmod,FvarFvarVar_FvarFvarVar_3rdDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using std::fmod;

  fvar<fvar<var> > x;
  x.val_.val_ = 3.0;
  x.val_.d_ = 1.0;
  fvar<fvar<var> > y;
  y.val_.val_ = 6.0;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = fmod(x,y);

  AVEC q = createAVEC(x.val_.val_,y.val_.val_);
  VEC r;
  a.d_.d_.grad(q,r);
  EXPECT_FLOAT_EQ(0, r[0]);
  EXPECT_FLOAT_EQ(0, r[1]);
}
TEST(AgradFwdFmod,FvarFvarVar_Double_3rdDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using std::fmod;

  fvar<fvar<var> > x;
  x.val_.val_ = 3.0;
  x.val_.d_ = 1.0;
  x.d_.val_ = 1.0;

  double y(6.0);

  fvar<fvar<var> > a = fmod(x,y);

  AVEC q = createAVEC(x.val_.val_);
  VEC r;
  a.d_.d_.grad(q,r);
  EXPECT_FLOAT_EQ(0, r[0]);
}
TEST(AgradFwdFmod,Double_FvarFvarVar_3rdDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using std::fmod;

  double x(3.0);
  fvar<fvar<var> > y;
  y.val_.val_ = 6.0;
  y.d_.val_ = 1.0;
  y.val_.d_ = 1.0;

  fvar<fvar<var> > a = fmod(x,y);

  AVEC q = createAVEC(y.val_.val_);
  VEC r;
  a.d_.d_.grad(q,r);
  EXPECT_FLOAT_EQ(0, r[0]);
}

struct fmod_fun {
  template <typename T0, typename T1>
  inline 
  typename boost::math::tools::promote_args<T0,T1>::type
  operator()(const T0 arg1,
             const T1 arg2) const {
    return fmod(arg1,arg2);
  }
};

TEST(AgradFwdFmod, nan) {
  fmod_fun fmod_;
  test_nan(fmod_,3.0,5.0,false);
}
