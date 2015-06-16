#include <stan/math/rev/scal/fun/Phi.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <gtest/gtest.h>
#include <test/unit/math/rev/scal/fun/nan_util.hpp>

TEST(AgradRev, Phi) {
  using stan::math::var;
  using std::exp;

  std::vector<double> y_values;
  y_values.push_back(0.0);
  y_values.push_back(0.9);
  y_values.push_back(-5.0);
  y_values.push_back(-27.5);
  y_values.push_back(27.5);

  // d/dy = exp(normal_log(value_of(y), 0.0, 1.0))
  std::vector<double> dy_values;
  dy_values.push_back(0.3989423);
  dy_values.push_back(0.2660852);
  dy_values.push_back(1.4867195e-06);
  dy_values.push_back(0);
  dy_values.push_back(0);

  for (size_t i = 0; i < y_values.size(); i++) {
    var y, phi_y;
    AVEC x;
    VEC dy;
    y = y_values[i];
    phi_y = stan::math::Phi(y);
    x = createAVEC(y);
    phi_y.grad(x,dy);
    EXPECT_FLOAT_EQ(stan::math::Phi(y.val()), phi_y.val());
    EXPECT_FLOAT_EQ(dy_values[i], dy[0])
      << "y = " << y;
  }
}

// tests calculating using R 3.0.2 Snow Leopard build (6558)
TEST(AgradRev, PhiTails) {
  using stan::math::Phi;
  using stan::math::var;

  EXPECT_EQ(0, Phi(var(-40)).val());

  EXPECT_FLOAT_EQ(1, 4.60535300958196e-308 / Phi(var(-37.5)).val());
  EXPECT_FLOAT_EQ(1, 5.72557122252458e-300 / Phi(var(-37)).val());
  EXPECT_FLOAT_EQ(1, 5.54472571307484e-292 / Phi(var(-36.5)).val());
  EXPECT_FLOAT_EQ(1, 4.18262406579728e-284 / Phi(var(-36)).val());
  EXPECT_FLOAT_EQ(1, 2.45769154066194e-276 / Phi(var(-35.5)).val());
  EXPECT_FLOAT_EQ(1, 1.12491070647241e-268 / Phi(var(-35)).val());
  EXPECT_FLOAT_EQ(1, 4.01072896657726e-261 / Phi(var(-34.5)).val());
  EXPECT_FLOAT_EQ(1, 1.11389878557438e-253 / Phi(var(-34)).val());
  EXPECT_FLOAT_EQ(1, 2.40983869512039e-246 / Phi(var(-33.5)).val());
  EXPECT_FLOAT_EQ(1, 4.06118562091586e-239 / Phi(var(-33)).val());
  EXPECT_FLOAT_EQ(1, 5.33142435967881e-232 / Phi(var(-32.5)).val());
  EXPECT_FLOAT_EQ(1, 5.4520806035124e-225 / Phi(var(-32)).val());
  EXPECT_FLOAT_EQ(1, 4.34323260103177e-218 / Phi(var(-31.5)).val());
  EXPECT_FLOAT_EQ(1, 2.6952500812005e-211 / Phi(var(-31)).val());
  EXPECT_FLOAT_EQ(1, 1.30293791317808e-204 / Phi(var(-30.5)).val());
  EXPECT_FLOAT_EQ(1, 4.90671392714819e-198 / Phi(var(-30)).val());
  EXPECT_FLOAT_EQ(1, 1.43947455222918e-191 / Phi(var(-29.5)).val());
  EXPECT_FLOAT_EQ(1, 3.28978526670438e-185 / Phi(var(-29)).val());
  EXPECT_FLOAT_EQ(1, 5.85714125380634e-179 / Phi(var(-28.5)).val());
  EXPECT_FLOAT_EQ(1, 8.12386946965943e-173 / Phi(var(-28)).val());
  EXPECT_FLOAT_EQ(1, 8.77817055687808e-167 / Phi(var(-27.5)).val());
  EXPECT_FLOAT_EQ(1, 7.38948100688502e-161 / Phi(var(-27)).val());
  EXPECT_FLOAT_EQ(1, 4.84616266030332e-155 / Phi(var(-26.5)).val());
  EXPECT_FLOAT_EQ(1, 2.47606331550339e-149 / Phi(var(-26)).val());
  EXPECT_FLOAT_EQ(1, 9.85623651896393e-144 / Phi(var(-25.5)).val());
  EXPECT_FLOAT_EQ(1, 3.05669670638256e-138 / Phi(var(-25)).val());
  EXPECT_FLOAT_EQ(1, 7.38570686148941e-133 / Phi(var(-24.5)).val());
  EXPECT_FLOAT_EQ(1, 1.3903921185497e-127 / Phi(var(-24)).val());
  EXPECT_FLOAT_EQ(1, 2.03936756324998e-122 / Phi(var(-23.5)).val());
  EXPECT_FLOAT_EQ(1, 2.33063700622065e-117 / Phi(var(-23)).val());
  EXPECT_FLOAT_EQ(1, 2.07531079906636e-112 / Phi(var(-22.5)).val());
  EXPECT_FLOAT_EQ(1, 1.43989243514508e-107 / Phi(var(-22)).val());
  EXPECT_FLOAT_EQ(1, 7.78439707718263e-103 / Phi(var(-21.5)).val());
  EXPECT_FLOAT_EQ(1, 3.27927801897904e-98 / Phi(var(-21)).val());
  EXPECT_FLOAT_EQ(1, 1.0764673258791e-93 / Phi(var(-20.5)).val());
  EXPECT_FLOAT_EQ(1, 2.75362411860623e-89 / Phi(var(-20)).val());
  EXPECT_FLOAT_EQ(1, 5.48911547566041e-85 / Phi(var(-19.5)).val());
  EXPECT_FLOAT_EQ(1, 8.52722395263098e-81 / Phi(var(-19)).val());
  EXPECT_FLOAT_EQ(1, 1.03236986895633e-76 / Phi(var(-18.5)).val());
  EXPECT_FLOAT_EQ(1, 9.74094891893715e-73 / Phi(var(-18)).val());
  EXPECT_FLOAT_EQ(1, 7.16345876623504e-69 / Phi(var(-17.5)).val());
  EXPECT_FLOAT_EQ(1, 4.10599620209891e-65 / Phi(var(-17)).val());
  EXPECT_FLOAT_EQ(1, 1.83446300316473e-61 / Phi(var(-16.5)).val());
  EXPECT_FLOAT_EQ(1, 6.38875440053809e-58 / Phi(var(-16)).val());
  EXPECT_FLOAT_EQ(1, 1.73446079179387e-54 / Phi(var(-15.5)).val());
  EXPECT_FLOAT_EQ(1, 3.67096619931275e-51 / Phi(var(-15)).val());
  EXPECT_FLOAT_EQ(1, 6.05749476441522e-48 / Phi(var(-14.5)).val());
  EXPECT_FLOAT_EQ(1, 7.7935368191928e-45 / Phi(var(-14)).val());
  EXPECT_FLOAT_EQ(1, 7.81880730565789e-42 / Phi(var(-13.5)).val());
  EXPECT_FLOAT_EQ(1, 6.11716439954988e-39 / Phi(var(-13)).val());
  EXPECT_FLOAT_EQ(1, 3.73256429887771e-36 / Phi(var(-12.5)).val());
  EXPECT_FLOAT_EQ(1, 1.77648211207768e-33 / Phi(var(-12)).val());
  EXPECT_FLOAT_EQ(1, 6.59577144611367e-31 / Phi(var(-11.5)).val());
  EXPECT_FLOAT_EQ(1, 1.91065957449868e-28 / Phi(var(-11)).val());
  EXPECT_FLOAT_EQ(1, 4.31900631780923e-26 / Phi(var(-10.5)).val());
  EXPECT_FLOAT_EQ(1, 7.61985302416053e-24 / Phi(var(-10)).val());
  EXPECT_FLOAT_EQ(1, 1.04945150753626e-21 / Phi(var(-9.5)).val());
  EXPECT_FLOAT_EQ(1, 1.12858840595384e-19 / Phi(var(-9)).val());
  EXPECT_FLOAT_EQ(1, 9.47953482220332e-18 / Phi(var(-8.5)).val());
  EXPECT_FLOAT_EQ(1, 6.22096057427178e-16 / Phi(var(-8)).val());
  EXPECT_FLOAT_EQ(1, 3.1908916729109e-14 / Phi(var(-7.5)).val());
  EXPECT_FLOAT_EQ(1, 1.27981254388584e-12 / Phi(var(-7)).val());
  EXPECT_FLOAT_EQ(1, 4.01600058385912e-11 / Phi(var(-6.5)).val());
  EXPECT_FLOAT_EQ(1, 9.86587645037698e-10 / Phi(var(-6)).val());
  EXPECT_FLOAT_EQ(1, 1.89895624658877e-08 / Phi(var(-5.5)).val());
  EXPECT_FLOAT_EQ(1, 2.86651571879194e-07 / Phi(var(-5)).val());
  EXPECT_FLOAT_EQ(1, 3.39767312473006e-06 / Phi(var(-4.5)).val());
  EXPECT_FLOAT_EQ(1, 3.16712418331199e-05 / Phi(var(-4)).val());
  EXPECT_FLOAT_EQ(1, 0.000232629079035525 / Phi(var(-3.5)).val());
  EXPECT_FLOAT_EQ(1, 0.00134989803163009 / Phi(var(-3)).val());
  EXPECT_FLOAT_EQ(1, 0.00620966532577613 / Phi(var(-2.5)).val());
  EXPECT_FLOAT_EQ(1, 0.0227501319481792 / Phi(var(-2)).val());
  EXPECT_FLOAT_EQ(1, 0.0668072012688581 / Phi(var(-1.5)).val());
  EXPECT_FLOAT_EQ(1, 0.158655253931457 / Phi(var(-1)).val());
  EXPECT_FLOAT_EQ(1, 0.308537538725987 / Phi(var(-0.5)).val());
  EXPECT_FLOAT_EQ(1, 0.5 / Phi(var(0)).val());
  EXPECT_FLOAT_EQ(1, 0.691462461274013 / Phi(var(0.5)).val());
  EXPECT_FLOAT_EQ(1, 0.841344746068543 / Phi(var(1)).val());
  EXPECT_FLOAT_EQ(1, 0.933192798731142 / Phi(var(1.5)).val());
  EXPECT_FLOAT_EQ(1, 0.977249868051821 / Phi(var(2)).val());
  EXPECT_FLOAT_EQ(1, 0.993790334674224 / Phi(var(2.5)).val());
  EXPECT_FLOAT_EQ(1, 0.99865010196837 / Phi(var(3)).val());
  EXPECT_FLOAT_EQ(1, 0.999767370920964 / Phi(var(3.5)).val());
  EXPECT_FLOAT_EQ(1, 0.999968328758167 / Phi(var(4)).val());
  EXPECT_FLOAT_EQ(1, 0.999996602326875 / Phi(var(4.5)).val());
  EXPECT_FLOAT_EQ(1, 0.999999713348428 / Phi(var(5)).val());
  EXPECT_FLOAT_EQ(1, 0.999999981010438 / Phi(var(5.5)).val());
  EXPECT_FLOAT_EQ(1, 0.999999999013412 / Phi(var(6)).val());
  EXPECT_FLOAT_EQ(1, 0.99999999995984 / Phi(var(6.5)).val());
  EXPECT_FLOAT_EQ(1, 0.99999999999872 / Phi(var(7)).val());
  EXPECT_FLOAT_EQ(1, 0.999999999999968 / Phi(var(7.5)).val());
  EXPECT_FLOAT_EQ(1, 0.999999999999999 / Phi(var(8)).val());
  EXPECT_FLOAT_EQ(1, 1 / Phi(var(8.5)).val());
  EXPECT_FLOAT_EQ(1, 1 / Phi(var(9)).val());
  EXPECT_FLOAT_EQ(1, 1 / Phi(var(9.5)).val());
  EXPECT_FLOAT_EQ(1, 1 / Phi(var(10)).val());
}

struct Phi_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return Phi(arg1);
  }
};

TEST(AgradRev,Phi_NaN) {
  Phi_fun Phi_;
  test_nan(Phi_,true,false);
}

