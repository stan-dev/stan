#include <gtest/gtest.h>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <test/unit/math/mix/scal/fun/nan_util.hpp>
#include <stan/math/fwd/scal/fun/fmod.hpp>
#include <stan/math/rev/scal/fun/fmod.hpp>
#include <stan/math/fwd/scal/fun/floor.hpp>
#include <stan/math/rev/scal/fun/floor.hpp>
#include <stan/math/fwd/scal/fun/value_of.hpp>
#include <stan/math/rev/scal/fun/value_of.hpp>


TEST(AgradFwdFmod,FvarVar_FvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
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
  using stan::math::fvar;
  using stan::math::var;
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
  using stan::math::fvar;
  using stan::math::var;
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
  using stan::math::fvar;
  using stan::math::var;
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
  using stan::math::fvar;
  using stan::math::var;
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
  using stan::math::fvar;
  using stan::math::var;
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

TEST(AgradFwdFmod,FvarFvarVar_FvarFvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
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
  using stan::math::fvar;
  using stan::math::var;
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
  using stan::math::fvar;
  using stan::math::var;
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
  using stan::math::fvar;
  using stan::math::var;
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
  using stan::math::fvar;
  using stan::math::var;
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
  using stan::math::fvar;
  using stan::math::var;
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
  using stan::math::fvar;
  using stan::math::var;
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
  using stan::math::fvar;
  using stan::math::var;
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
  using stan::math::fvar;
  using stan::math::var;
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
  using stan::math::fvar;
  using stan::math::var;
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
  test_nan_mix(fmod_,3.0,5.0,false);
}
