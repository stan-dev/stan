#include <gtest/gtest.h>
#include <stan/math/fwd/scal/fun/Phi.hpp>
#include <stan/math/prim/scal/fun/Phi.hpp>
#include <stan/math/prim/scal/prob/normal_log.hpp>
#include <test/unit/math/fwd/scal/fun/nan_util.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/fwd/scal/fun/exp.hpp>

TEST(AgradFwdPhi,Fvar) {
  using stan::math::fvar;
  using stan::math::Phi;
  fvar<double> x = 1.0;
  x.d_ = 1.0;
  
  fvar<double> Phi_x = Phi(x);

  EXPECT_FLOAT_EQ(Phi(1.0), Phi_x.val_);
  EXPECT_FLOAT_EQ(exp(stan::math::normal_log<false>(1.0,0.0,1.0)),
                  Phi_x.d_);
}
TEST(AgradFwdPhi, FvarDerivUnderOverFlow) {
  using stan::math::fvar;
  fvar<double> x = -27.5;
  x.d_ = 1.0;
  fvar<double> Phi_x = Phi(x);
  EXPECT_FLOAT_EQ(0, Phi_x.d_);

  fvar<double> y = 27.5;
  y.d_ = 1.0;
  fvar<double> Phi_y = Phi(y);
  EXPECT_FLOAT_EQ(0, Phi_y.d_);
}

TEST(AgradFwdPhi, FvarFvarDouble) {
  using stan::math::fvar;
  using stan::math::Phi;

  fvar<fvar<double> > x;
  x.val_.val_ = 1.0;
  x.val_.d_ = 1.0;

  fvar<fvar<double> > a = Phi(x);

  EXPECT_FLOAT_EQ(Phi(1.0), a.val_.val_);
  EXPECT_FLOAT_EQ(exp(stan::math::normal_log<false>(1.0,0.0,1.0)), a.val_.d_);
  EXPECT_FLOAT_EQ(0, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);

  fvar<fvar<double> > y;
  y.val_.val_ = 1.0;
  y.d_.val_ = 1.0;

  a = Phi(y);
  EXPECT_FLOAT_EQ(Phi(1.0), a.val_.val_);
  EXPECT_FLOAT_EQ(0, a.val_.d_);
  EXPECT_FLOAT_EQ(exp(stan::math::normal_log<false>(1.0,0.0,1.0)), a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);
}

double fun_value_of(double x) { return x; }
// inefficient calls by value because neither const& nor & compile
double fun_value_of(stan::math::fvar<double> x) {
  return x.val();
}
double fun_value_of(stan::math::fvar<stan::math::fvar<double> > x) {
  return x.val().val();
}

// tests calculating using R 3.0.2 Snow Leopard build (6558)
template <typename T>
void test_tails() {
  using stan::math::Phi;

  EXPECT_EQ(0, fun_value_of(Phi(T(-40))));

  EXPECT_FLOAT_EQ(1, 4.60535300958196e-308 / fun_value_of(Phi(T(-37.5))));
  EXPECT_FLOAT_EQ(1, 5.72557122252458e-300 / fun_value_of(Phi(T(-37))));
  EXPECT_FLOAT_EQ(1, 5.54472571307484e-292 / fun_value_of(Phi(T(-36.5))));
  EXPECT_FLOAT_EQ(1, 4.18262406579728e-284 / fun_value_of(Phi(T(-36))));
  EXPECT_FLOAT_EQ(1, 2.45769154066194e-276 / fun_value_of(Phi(T(-35.5))));
  EXPECT_FLOAT_EQ(1, 1.12491070647241e-268 / fun_value_of(Phi(T(-35))));
  EXPECT_FLOAT_EQ(1, 4.01072896657726e-261 / fun_value_of(Phi(T(-34.5))));
  EXPECT_FLOAT_EQ(1, 1.11389878557438e-253 / fun_value_of(Phi(T(-34))));
  EXPECT_FLOAT_EQ(1, 2.40983869512039e-246 / fun_value_of(Phi(T(-33.5))));
  EXPECT_FLOAT_EQ(1, 4.06118562091586e-239 / fun_value_of(Phi(T(-33))));
  EXPECT_FLOAT_EQ(1, 5.33142435967881e-232 / fun_value_of(Phi(T(-32.5))));
  EXPECT_FLOAT_EQ(1, 5.4520806035124e-225 / fun_value_of(Phi(T(-32))));
  EXPECT_FLOAT_EQ(1, 4.34323260103177e-218 / fun_value_of(Phi(T(-31.5))));
  EXPECT_FLOAT_EQ(1, 2.6952500812005e-211 / fun_value_of(Phi(T(-31))));
  EXPECT_FLOAT_EQ(1, 1.30293791317808e-204 / fun_value_of(Phi(T(-30.5))));
  EXPECT_FLOAT_EQ(1, 4.90671392714819e-198 / fun_value_of(Phi(T(-30))));
  EXPECT_FLOAT_EQ(1, 1.43947455222918e-191 / fun_value_of(Phi(T(-29.5))));
  EXPECT_FLOAT_EQ(1, 3.28978526670438e-185 / fun_value_of(Phi(T(-29))));
  EXPECT_FLOAT_EQ(1, 5.85714125380634e-179 / fun_value_of(Phi(T(-28.5))));
  EXPECT_FLOAT_EQ(1, 8.12386946965943e-173 / fun_value_of(Phi(T(-28))));
  EXPECT_FLOAT_EQ(1, 8.77817055687808e-167 / fun_value_of(Phi(T(-27.5))));
  EXPECT_FLOAT_EQ(1, 7.38948100688502e-161 / fun_value_of(Phi(T(-27))));
  EXPECT_FLOAT_EQ(1, 4.84616266030332e-155 / fun_value_of(Phi(T(-26.5))));
  EXPECT_FLOAT_EQ(1, 2.47606331550339e-149 / fun_value_of(Phi(T(-26))));
  EXPECT_FLOAT_EQ(1, 9.85623651896393e-144 / fun_value_of(Phi(T(-25.5))));
  EXPECT_FLOAT_EQ(1, 3.05669670638256e-138 / fun_value_of(Phi(T(-25))));
  EXPECT_FLOAT_EQ(1, 7.38570686148941e-133 / fun_value_of(Phi(T(-24.5))));
  EXPECT_FLOAT_EQ(1, 1.3903921185497e-127 / fun_value_of(Phi(T(-24))));
  EXPECT_FLOAT_EQ(1, 2.03936756324998e-122 / fun_value_of(Phi(T(-23.5))));
  EXPECT_FLOAT_EQ(1, 2.33063700622065e-117 / fun_value_of(Phi(T(-23))));
  EXPECT_FLOAT_EQ(1, 2.07531079906636e-112 / fun_value_of(Phi(T(-22.5))));
  EXPECT_FLOAT_EQ(1, 1.43989243514508e-107 / fun_value_of(Phi(T(-22))));
  EXPECT_FLOAT_EQ(1, 7.78439707718263e-103 / fun_value_of(Phi(T(-21.5))));
  EXPECT_FLOAT_EQ(1, 3.27927801897904e-98 / fun_value_of(Phi(T(-21))));
  EXPECT_FLOAT_EQ(1, 1.0764673258791e-93 / fun_value_of(Phi(T(-20.5))));
  EXPECT_FLOAT_EQ(1, 2.75362411860623e-89 / fun_value_of(Phi(T(-20))));
  EXPECT_FLOAT_EQ(1, 5.48911547566041e-85 / fun_value_of(Phi(T(-19.5))));
  EXPECT_FLOAT_EQ(1, 8.52722395263098e-81 / fun_value_of(Phi(T(-19))));
  EXPECT_FLOAT_EQ(1, 1.03236986895633e-76 / fun_value_of(Phi(T(-18.5))));
  EXPECT_FLOAT_EQ(1, 9.74094891893715e-73 / fun_value_of(Phi(T(-18))));
  EXPECT_FLOAT_EQ(1, 7.16345876623504e-69 / fun_value_of(Phi(T(-17.5))));
  EXPECT_FLOAT_EQ(1, 4.10599620209891e-65 / fun_value_of(Phi(T(-17))));
  EXPECT_FLOAT_EQ(1, 1.83446300316473e-61 / fun_value_of(Phi(T(-16.5))));
  EXPECT_FLOAT_EQ(1, 6.38875440053809e-58 / fun_value_of(Phi(T(-16))));
  EXPECT_FLOAT_EQ(1, 1.73446079179387e-54 / fun_value_of(Phi(T(-15.5))));
  EXPECT_FLOAT_EQ(1, 3.67096619931275e-51 / fun_value_of(Phi(T(-15))));
  EXPECT_FLOAT_EQ(1, 6.05749476441522e-48 / fun_value_of(Phi(T(-14.5))));
  EXPECT_FLOAT_EQ(1, 7.7935368191928e-45 / fun_value_of(Phi(T(-14))));
  EXPECT_FLOAT_EQ(1, 7.81880730565789e-42 / fun_value_of(Phi(T(-13.5))));
  EXPECT_FLOAT_EQ(1, 6.11716439954988e-39 / fun_value_of(Phi(T(-13))));
  EXPECT_FLOAT_EQ(1, 3.73256429887771e-36 / fun_value_of(Phi(T(-12.5))));
  EXPECT_FLOAT_EQ(1, 1.77648211207768e-33 / fun_value_of(Phi(T(-12))));
  EXPECT_FLOAT_EQ(1, 6.59577144611367e-31 / fun_value_of(Phi(T(-11.5))));
  EXPECT_FLOAT_EQ(1, 1.91065957449868e-28 / fun_value_of(Phi(T(-11))));
  EXPECT_FLOAT_EQ(1, 4.31900631780923e-26 / fun_value_of(Phi(T(-10.5))));
  EXPECT_FLOAT_EQ(1, 7.61985302416053e-24 / fun_value_of(Phi(T(-10))));
  EXPECT_FLOAT_EQ(1, 1.04945150753626e-21 / fun_value_of(Phi(T(-9.5))));
  EXPECT_FLOAT_EQ(1, 1.12858840595384e-19 / fun_value_of(Phi(T(-9))));
  EXPECT_FLOAT_EQ(1, 9.47953482220332e-18 / fun_value_of(Phi(T(-8.5))));
  EXPECT_FLOAT_EQ(1, 6.22096057427178e-16 / fun_value_of(Phi(T(-8))));
  EXPECT_FLOAT_EQ(1, 3.1908916729109e-14 / fun_value_of(Phi(T(-7.5))));
  EXPECT_FLOAT_EQ(1, 1.27981254388584e-12 / fun_value_of(Phi(T(-7))));
  EXPECT_FLOAT_EQ(1, 4.01600058385912e-11 / fun_value_of(Phi(T(-6.5))));
  EXPECT_FLOAT_EQ(1, 9.86587645037698e-10 / fun_value_of(Phi(T(-6))));
  EXPECT_FLOAT_EQ(1, 1.89895624658877e-08 / fun_value_of(Phi(T(-5.5))));
  EXPECT_FLOAT_EQ(1, 2.86651571879194e-07 / fun_value_of(Phi(T(-5))));
  EXPECT_FLOAT_EQ(1, 3.39767312473006e-06 / fun_value_of(Phi(T(-4.5))));
  EXPECT_FLOAT_EQ(1, 3.16712418331199e-05 / fun_value_of(Phi(T(-4))));
  EXPECT_FLOAT_EQ(1, 0.000232629079035525 / fun_value_of(Phi(T(-3.5))));
  EXPECT_FLOAT_EQ(1, 0.00134989803163009 / fun_value_of(Phi(T(-3))));
  EXPECT_FLOAT_EQ(1, 0.00620966532577613 / fun_value_of(Phi(T(-2.5))));
  EXPECT_FLOAT_EQ(1, 0.0227501319481792 / fun_value_of(Phi(T(-2))));
  EXPECT_FLOAT_EQ(1, 0.0668072012688581 / fun_value_of(Phi(T(-1.5))));
  EXPECT_FLOAT_EQ(1, 0.158655253931457 / fun_value_of(Phi(T(-1))));
  EXPECT_FLOAT_EQ(1, 0.308537538725987 / fun_value_of(Phi(T(-0.5))));
  EXPECT_FLOAT_EQ(1, 0.5 / fun_value_of(Phi(T(0))));
  EXPECT_FLOAT_EQ(1, 0.691462461274013 / fun_value_of(Phi(T(0.5))));
  EXPECT_FLOAT_EQ(1, 0.841344746068543 / fun_value_of(Phi(T(1))));
  EXPECT_FLOAT_EQ(1, 0.933192798731142 / fun_value_of(Phi(T(1.5))));
  EXPECT_FLOAT_EQ(1, 0.977249868051821 / fun_value_of(Phi(T(2))));
  EXPECT_FLOAT_EQ(1, 0.993790334674224 / fun_value_of(Phi(T(2.5))));
  EXPECT_FLOAT_EQ(1, 0.99865010196837 / fun_value_of(Phi(T(3))));
  EXPECT_FLOAT_EQ(1, 0.999767370920964 / fun_value_of(Phi(T(3.5))));
  EXPECT_FLOAT_EQ(1, 0.999968328758167 / fun_value_of(Phi(T(4))));
  EXPECT_FLOAT_EQ(1, 0.999996602326875 / fun_value_of(Phi(T(4.5))));
  EXPECT_FLOAT_EQ(1, 0.999999713348428 / fun_value_of(Phi(T(5))));
  EXPECT_FLOAT_EQ(1, 0.999999981010438 / fun_value_of(Phi(T(5.5))));
  EXPECT_FLOAT_EQ(1, 0.999999999013412 / fun_value_of(Phi(T(6))));
  EXPECT_FLOAT_EQ(1, 0.99999999995984 / fun_value_of(Phi(T(6.5))));
  EXPECT_FLOAT_EQ(1, 0.99999999999872 / fun_value_of(Phi(T(7))));
  EXPECT_FLOAT_EQ(1, 0.999999999999968 / fun_value_of(Phi(T(7.5))));
  EXPECT_FLOAT_EQ(1, 0.999999999999999 / fun_value_of(Phi(T(8))));
  EXPECT_FLOAT_EQ(1, 1 / fun_value_of(Phi(T(8.5))));
  EXPECT_FLOAT_EQ(1, 1 / fun_value_of(Phi(T(9))));
  EXPECT_FLOAT_EQ(1, 1 / fun_value_of(Phi(T(9.5))));
  EXPECT_FLOAT_EQ(1, 1 / fun_value_of(Phi(T(10))));
}
TEST(AgradFwdPhi, PhiTails) {
  using stan::math::fvar;
  test_tails<fvar<double> >();
  test_tails<fvar<fvar<double> > >();
}

struct Phi_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return Phi(arg1);
  }
};

TEST(AgradFwdPhi,Phi_NaN) {
  Phi_fun Phi_;
  test_nan_fwd(Phi_,true);
}
