#include <gtest/gtest.h>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <test/unit/math/mix/scal/fun/nan_util.hpp>
#include <stan/math/fwd/scal/fun/fdim.hpp>
#include <stan/math/rev/scal/fun/fdim.hpp>
#include <stan/math/fwd/scal/fun/floor.hpp>
#include <stan/math/rev/scal/fun/floor.hpp>



TEST(AgradFwdFdim,FvarVar_FvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::fdim;
  using std::floor;
  using std::isnan;

  fvar<var> x(2.5,1.3);
  fvar<var> z(1.5,1.0);
  fvar<var> a = fdim(x,z);

  EXPECT_FLOAT_EQ(fdim(2.5,1.5), a.val_.val());
  isnan(a.d_.val());

  AVEC y = createAVEC(x.val_,z.val_);
  VEC g;
  a.val_.grad(y,g);
  isnan(g[0]);
  isnan(g[1]);
}

TEST(AgradFwdFdim,FvarVar_double_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::fdim;
  using std::floor;
  using std::isnan;

  fvar<var> x(2.5,1.3);
  double z(1.5);
  fvar<var> a = fdim(x,z);

  EXPECT_FLOAT_EQ(fdim(2.5,1.5), a.val_.val());
  isnan(a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.val_.grad(y,g);
  isnan(g[0]);
}

TEST(AgradFwdFdim,Double_FvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::fdim;
  using std::floor;
  using std::isnan;

  double x(2.5);
  fvar<var> z(1.5,1.0);
  fvar<var> a = fdim(x,z);

  EXPECT_FLOAT_EQ(fdim(2.5,1.5), a.val_.val());
  isnan(a.d_.val());

  AVEC y = createAVEC(z.val_);
  VEC g;
  a.val_.grad(y,g);
  isnan(g[0]);
}
TEST(AgradFwdFdim,FvarVar_FvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::fdim;
  using std::floor;
  using std::isnan;

  fvar<var> x(2.5,1.3);
  fvar<var> z(1.5,1.0);
  fvar<var> a = fdim(x,z);

  AVEC y = createAVEC(x.val_,z.val_);
  VEC g;
  a.d_.grad(y,g);
  isnan(g[0]);
  isnan(g[1]);
}

TEST(AgradFwdFdim,FvarVar_double_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::fdim;
  using std::floor;
  using std::isnan;

  fvar<var> x(2.5,1.3);
  double z(1.5);
  fvar<var> a = fdim(x,z);

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.d_.grad(y,g);
  isnan(g[0]);
}

TEST(AgradFwdFdim,Double_FvarVar_2nd_Deriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::fdim;
  using std::floor;
  using std::isnan;

  double x(2.5);
  fvar<var> z(1.5,1.0);
  fvar<var> a = fdim(x,z);

  AVEC y = createAVEC(z.val_);
  VEC g;
  a.d_.grad(y,g);
  isnan(g[0]);
}


TEST(AgradFwdFdim,FvarFvarVar_FvarFvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::fdim;
  using std::floor;

  fvar<fvar<var> > x;
  x.val_.val_ = 2.5;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = fdim(x,y);

  EXPECT_FLOAT_EQ(fdim(2.5,1.5), a.val_.val_.val());
  EXPECT_FLOAT_EQ(1, a.val_.d_.val());
  EXPECT_FLOAT_EQ(-1, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_, y.val_.val_);
  VEC g;
  a.val_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(1.0, g[0]);
  EXPECT_FLOAT_EQ(-1.0, g[1]);
}

TEST(AgradFwdFdim,FvarFvarVar_double_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::fdim;
  using std::floor;

  fvar<fvar<var> > x;
  x.val_.val_ = 2.5;
  x.val_.d_ = 1.0;

  double y(1.5);

  fvar<fvar<var> > a = fdim(x,y);

  EXPECT_FLOAT_EQ(fdim(2.5,1.5), a.val_.val_.val());
  EXPECT_FLOAT_EQ(1, a.val_.d_.val());
  EXPECT_FLOAT_EQ(0, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(1.0, g[0]);
}

TEST(AgradFwdFdim,Double_FvarFvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::fdim;
  using std::floor;

  double x(2.5);

  fvar<fvar<var> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = fdim(x,y);

  EXPECT_FLOAT_EQ(fdim(2.5,1.5), a.val_.val_.val());
  EXPECT_FLOAT_EQ(0, a.val_.d_.val());
  EXPECT_FLOAT_EQ(-1, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(y.val_.val_);
  VEC g;
  a.val_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(-1.0, g[0]);
}

TEST(AgradFwdFdim,FvarFvarVar_FvarFvarVar_2ndDeriv_x) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::fdim;
  using std::floor;

  fvar<fvar<var> > x;
  x.val_.val_ = 2.5;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = fdim(x,y);

  EXPECT_FLOAT_EQ(fdim(2.5,1.5), a.val_.val_.val());
  EXPECT_FLOAT_EQ(1, a.val_.d_.val());
  EXPECT_FLOAT_EQ(-1, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(0.0, g[0]);
}

TEST(AgradFwdFdim,FvarFvarVar_FvarFvarVar_2ndDeriv_y) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::fdim;
  using std::floor;

  fvar<fvar<var> > x;
  x.val_.val_ = 2.5;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = fdim(x,y);

  EXPECT_FLOAT_EQ(fdim(2.5,1.5), a.val_.val_.val());
  EXPECT_FLOAT_EQ(1, a.val_.d_.val());
  EXPECT_FLOAT_EQ(-1, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(y.val_.val_);
  VEC g;
  a.d_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(0.0, g[0]);
}
TEST(AgradFwdFdim,FvarFvarVar_double_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::fdim;
  using std::floor;

  fvar<fvar<var> > x;
  x.val_.val_ = 2.5;
  x.val_.d_ = 1.0;

  double y(1.5);

  fvar<fvar<var> > a = fdim(x,y);

  EXPECT_FLOAT_EQ(fdim(2.5,1.5), a.val_.val_.val());
  EXPECT_FLOAT_EQ(1, a.val_.d_.val());
  EXPECT_FLOAT_EQ(0, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(0.0, g[0]);
}

TEST(AgradFwdFdim,Double_FvarFvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::fdim;
  using std::floor;

  double x(2.5);

  fvar<fvar<var> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = fdim(x,y);

  EXPECT_FLOAT_EQ(fdim(2.5,1.5), a.val_.val_.val());
  EXPECT_FLOAT_EQ(0, a.val_.d_.val());
  EXPECT_FLOAT_EQ(-1, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(y.val_.val_);
  VEC g;
  a.d_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(0.0, g[0]);
}

TEST(AgradFwdFdim,FvarFvarVar_FvarFvarVar_3rdDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::fdim;
  using std::floor;

  fvar<fvar<var> > x;
  x.val_.val_ = 2.5;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = fdim(x,y);

  AVEC p = createAVEC(y.val_.val_);
  VEC g;
  a.d_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(0.0, g[0]);
}
TEST(AgradFwdFdim,FvarFvarVar_double_3rdDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::fdim;
  using std::floor;

  fvar<fvar<var> > x;
  x.val_.val_ = 2.5;
  x.val_.d_ = 1.0;
  x.d_.val_ = 1.0;

  double y(1.5);

  fvar<fvar<var> > a = fdim(x,y);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.d_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(0.0, g[0]);
}

TEST(AgradFwdFdim,Double_FvarFvarVar_3rdDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::fdim;
  using std::floor;

  double x(2.5);

  fvar<fvar<var> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 1.0;
  y.val_.d_ = 1.0;

  fvar<fvar<var> > a = fdim(x,y);

  AVEC p = createAVEC(y.val_.val_);
  VEC g;
  a.d_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(0.0, g[0]);
}

struct fdim_fun {
  template <typename T0, typename T1>
  inline 
  typename boost::math::tools::promote_args<T0,T1>::type
  operator()(const T0 arg1,
             const T1 arg2) const {
    return fdim(arg1,arg2);
  }
};

TEST(AgradFwdFdim, nan) {
  fdim_fun fdim_;
  test_nan_mix(fdim_,3.0,5.0,false);
}
