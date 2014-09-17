#include <stan/prob/distributions/univariate/continuous/normal.hpp>
#include <gtest/gtest.h>
#include <stan/agrad/rev.hpp>
#include <boost/math/distributions.hpp>
#include <boost/random/mersenne_twister.hpp>

TEST(ProbDistributionsNormal, cdf_tails) {
  using stan::agrad::var;
  using stan::prob::normal_cdf;

  EXPECT_FLOAT_EQ(1, 4.60535300958196e-308/normal_cdf(var(-37.5),0,1).val());
  EXPECT_FLOAT_EQ(1, 5.72557122252458e-300 / normal_cdf(var(-37),0,1).val());
  EXPECT_FLOAT_EQ(1, 5.54472571307484e-292 / normal_cdf(var(-36.5),0,1).val());
  EXPECT_FLOAT_EQ(1, 4.18262406579728e-284 / normal_cdf(var(-36),0,1).val());
  EXPECT_FLOAT_EQ(1, 2.45769154066194e-276 / normal_cdf(var(-35.5),0,1).val());
  EXPECT_FLOAT_EQ(1, 1.12491070647241e-268 / normal_cdf(var(-35),0,1).val());
  EXPECT_FLOAT_EQ(1, 4.01072896657726e-261 / normal_cdf(var(-34.5),0,1).val());
  EXPECT_FLOAT_EQ(1, 1.11389878557438e-253 / normal_cdf(var(-34),0,1).val());
  EXPECT_FLOAT_EQ(1, 2.40983869512039e-246 / normal_cdf(var(-33.5),0,1).val());
  EXPECT_FLOAT_EQ(1, 4.06118562091586e-239 / normal_cdf(var(-33),0,1).val());
  EXPECT_FLOAT_EQ(1, 5.33142435967881e-232 / normal_cdf(var(-32.5),0,1).val());
  EXPECT_FLOAT_EQ(1, 5.4520806035124e-225 / normal_cdf(var(-32),0,1).val());
  EXPECT_FLOAT_EQ(1, 4.34323260103177e-218 / normal_cdf(var(-31.5),0,1).val());
  EXPECT_FLOAT_EQ(1, 2.6952500812005e-211 / normal_cdf(var(-31),0,1).val());
  EXPECT_FLOAT_EQ(1, 1.30293791317808e-204 / normal_cdf(var(-30.5),0,1).val());
  EXPECT_FLOAT_EQ(1, 4.90671392714819e-198 / normal_cdf(var(-30),0,1).val());
  EXPECT_FLOAT_EQ(1, 1.43947455222918e-191 / normal_cdf(var(-29.5),0,1).val());
  EXPECT_FLOAT_EQ(1, 3.28978526670438e-185 / normal_cdf(var(-29),0,1).val());
  EXPECT_FLOAT_EQ(1, 5.85714125380634e-179 / normal_cdf(var(-28.5),0,1).val());
  EXPECT_FLOAT_EQ(1, 8.12386946965943e-173 / normal_cdf(var(-28),0,1).val());
  EXPECT_FLOAT_EQ(1, 8.77817055687808e-167 / normal_cdf(var(-27.5),0,1).val());
  EXPECT_FLOAT_EQ(1, 7.38948100688502e-161 / normal_cdf(var(-27),0,1).val());
  EXPECT_FLOAT_EQ(1, 4.84616266030332e-155 / normal_cdf(var(-26.5),0,1).val());
  EXPECT_FLOAT_EQ(1, 2.47606331550339e-149 / normal_cdf(var(-26),0,1).val());
  EXPECT_FLOAT_EQ(1, 9.85623651896393e-144 / normal_cdf(var(-25.5),0,1).val());
  EXPECT_FLOAT_EQ(1, 3.05669670638256e-138 / normal_cdf(var(-25),0,1).val());
  EXPECT_FLOAT_EQ(1, 7.38570686148941e-133 / normal_cdf(var(-24.5),0,1).val());
  EXPECT_FLOAT_EQ(1, 1.3903921185497e-127 / normal_cdf(var(-24),0,1).val());
  EXPECT_FLOAT_EQ(1, 2.03936756324998e-122 / normal_cdf(var(-23.5),0,1).val());
  EXPECT_FLOAT_EQ(1, 2.33063700622065e-117 / normal_cdf(var(-23),0,1).val());
  EXPECT_FLOAT_EQ(1, 2.07531079906636e-112 / normal_cdf(var(-22.5),0,1).val());
  EXPECT_FLOAT_EQ(1, 1.43989243514508e-107 / normal_cdf(var(-22),0,1).val());
  EXPECT_FLOAT_EQ(1, 7.78439707718263e-103 / normal_cdf(var(-21.5),0,1).val());
  EXPECT_FLOAT_EQ(1, 3.27927801897904e-98 / normal_cdf(var(-21),0,1).val());
  EXPECT_FLOAT_EQ(1, 1.0764673258791e-93 / normal_cdf(var(-20.5),0,1).val());
  EXPECT_FLOAT_EQ(1, 2.75362411860623e-89 / normal_cdf(var(-20),0,1).val());
  EXPECT_FLOAT_EQ(1, 5.48911547566041e-85 / normal_cdf(var(-19.5),0,1).val());
  EXPECT_FLOAT_EQ(1, 8.52722395263098e-81 / normal_cdf(var(-19),0,1).val());
  EXPECT_FLOAT_EQ(1, 1.03236986895633e-76 / normal_cdf(var(-18.5),0,1).val());
  EXPECT_FLOAT_EQ(1, 9.74094891893715e-73 / normal_cdf(var(-18),0,1).val());
  EXPECT_FLOAT_EQ(1, 7.16345876623504e-69 / normal_cdf(var(-17.5),0,1).val());
  EXPECT_FLOAT_EQ(1, 4.10599620209891e-65 / normal_cdf(var(-17),0,1).val());
  EXPECT_FLOAT_EQ(1, 1.83446300316473e-61 / normal_cdf(var(-16.5),0,1).val());
  EXPECT_FLOAT_EQ(1, 6.38875440053809e-58 / normal_cdf(var(-16),0,1).val());
  EXPECT_FLOAT_EQ(1, 1.73446079179387e-54 / normal_cdf(var(-15.5),0,1).val());
  EXPECT_FLOAT_EQ(1, 3.67096619931275e-51 / normal_cdf(var(-15),0,1).val());
  EXPECT_FLOAT_EQ(1, 6.05749476441522e-48 / normal_cdf(var(-14.5),0,1).val());
  EXPECT_FLOAT_EQ(1, 7.7935368191928e-45 / normal_cdf(var(-14),0,1).val());
  EXPECT_FLOAT_EQ(1, 7.81880730565789e-42 / normal_cdf(var(-13.5),0,1).val());
  EXPECT_FLOAT_EQ(1, 6.11716439954988e-39 / normal_cdf(var(-13),0,1).val());
  EXPECT_FLOAT_EQ(1, 3.73256429887771e-36 / normal_cdf(var(-12.5),0,1).val());
  EXPECT_FLOAT_EQ(1, 1.77648211207768e-33 / normal_cdf(var(-12),0,1).val());
 EXPECT_FLOAT_EQ(1, 6.59577144611367e-31 / normal_cdf(var(-11.5),0,1).val());
  EXPECT_FLOAT_EQ(1, 1.91065957449868e-28 / normal_cdf(var(-11),0,1).val());
  EXPECT_FLOAT_EQ(1, 4.31900631780923e-26 / normal_cdf(var(-10.5),0,1).val());
  EXPECT_FLOAT_EQ(1, 7.61985302416053e-24 / normal_cdf(var(-10),0,1).val());
  EXPECT_FLOAT_EQ(1, 1.04945150753626e-21 / normal_cdf(var(-9.5),0,1).val());
  EXPECT_FLOAT_EQ(1, 1.12858840595384e-19 / normal_cdf(var(-9),0,1).val());
  EXPECT_FLOAT_EQ(1, 9.47953482220332e-18 / normal_cdf(var(-8.5),0,1).val());
  EXPECT_FLOAT_EQ(1, 6.22096057427178e-16 / normal_cdf(var(-8),0,1).val());
  EXPECT_FLOAT_EQ(1, 3.1908916729109e-14 / normal_cdf(var(-7.5),0,1).val());
  EXPECT_FLOAT_EQ(1, 1.27981254388584e-12 / normal_cdf(var(-7),0,1).val());
  EXPECT_FLOAT_EQ(1, 4.01600058385912e-11 / normal_cdf(var(-6.5),0,1).val());
  EXPECT_FLOAT_EQ(1, 9.86587645037698e-10 / normal_cdf(var(-6),0,1).val());
  EXPECT_FLOAT_EQ(1, 1.89895624658877e-08 / normal_cdf(var(-5.5),0,1).val());
  EXPECT_FLOAT_EQ(1, 2.86651571879194e-07 / normal_cdf(var(-5),0,1).val());
  EXPECT_FLOAT_EQ(1, 3.39767312473006e-06 / normal_cdf(var(-4.5),0,1).val());
  EXPECT_FLOAT_EQ(1, 3.16712418331199e-05 / normal_cdf(var(-4),0,1).val());
  EXPECT_FLOAT_EQ(1, 0.000232629079035525 / normal_cdf(var(-3.5),0,1).val());
  EXPECT_FLOAT_EQ(1, 0.00134989803163009 / normal_cdf(var(-3),0,1).val());
  EXPECT_FLOAT_EQ(1, 0.00620966532577613 / normal_cdf(var(-2.5),0,1).val());
  EXPECT_FLOAT_EQ(1, 0.0227501319481792 / normal_cdf(var(-2),0,1).val());
  EXPECT_FLOAT_EQ(1, 0.0668072012688581 / normal_cdf(var(-1.5),0,1).val());
  EXPECT_FLOAT_EQ(1, 0.158655253931457 / normal_cdf(var(-1),0,1).val());
  EXPECT_FLOAT_EQ(1, 0.308537538725987 / normal_cdf(var(-0.5),0,1).val());
  EXPECT_FLOAT_EQ(1, 0.5 / normal_cdf(var(0),0,1).val());
  EXPECT_FLOAT_EQ(1, 0.691462461274013 / normal_cdf(var(0.5),0,1).val());
  EXPECT_FLOAT_EQ(1, 0.841344746068543 / normal_cdf(var(1),0,1).val());
  EXPECT_FLOAT_EQ(1, 0.933192798731142 / normal_cdf(var(1.5),0,1).val());
  EXPECT_FLOAT_EQ(1, 0.977249868051821 / normal_cdf(var(2),0,1).val());
  EXPECT_FLOAT_EQ(1, 0.993790334674224 / normal_cdf(var(2.5),0,1).val());
  EXPECT_FLOAT_EQ(1, 0.99865010196837 / normal_cdf(var(3),0,1).val());
  EXPECT_FLOAT_EQ(1, 0.999767370920964 / normal_cdf(var(3.5),0,1).val());
  EXPECT_FLOAT_EQ(1, 0.999968328758167 / normal_cdf(var(4),0,1).val());
  EXPECT_FLOAT_EQ(1, 0.999996602326875 / normal_cdf(var(4.5),0,1).val());
  EXPECT_FLOAT_EQ(1, 0.999999713348428 / normal_cdf(var(5),0,1).val());
  EXPECT_FLOAT_EQ(1, 0.999999981010438 / normal_cdf(var(5.5),0,1).val());
  EXPECT_FLOAT_EQ(1, 0.999999999013412 / normal_cdf(var(6),0,1).val());
  EXPECT_FLOAT_EQ(1, 0.99999999995984 / normal_cdf(var(6.5),0,1).val());
  EXPECT_FLOAT_EQ(1, 0.99999999999872 / normal_cdf(var(7),0,1).val());
  EXPECT_FLOAT_EQ(1, 0.999999999999968 / normal_cdf(var(7.5),0,1).val());
  EXPECT_FLOAT_EQ(1, 0.999999999999999 / normal_cdf(var(8),0,1).val());
  EXPECT_FLOAT_EQ(1, 1 / normal_cdf(var(8.5),0,1).val());
  EXPECT_FLOAT_EQ(1, 1 / normal_cdf(var(9),0,1).val());
  EXPECT_FLOAT_EQ(1, 1 / normal_cdf(var(9.5),0,1).val());
  EXPECT_FLOAT_EQ(1, 1 / normal_cdf(var(10),0,1).val()); 
}
TEST(ProbDistributionsNormal, cdf_log_tails) {
  using stan::agrad::var;
  using stan::prob::normal_cdf_log;
  using std::exp;

  EXPECT_FLOAT_EQ(1,4.60535300958196e-308/ exp(normal_cdf_log(var(-37.5),0,1).val()));
  EXPECT_FLOAT_EQ(1,5.72557122252458e-300 / exp(normal_cdf_log(var(-37),0,1).val()));
  EXPECT_FLOAT_EQ(1,5.54472571307484e-292 / exp(normal_cdf_log(var(-36.5),0,1).val()));
  EXPECT_FLOAT_EQ(1,4.18262406579728e-284 / exp(normal_cdf_log(var(-36),0,1).val()));
  EXPECT_FLOAT_EQ(1,2.45769154066194e-276 / exp(normal_cdf_log(var(-35.5),0,1).val()));
  EXPECT_FLOAT_EQ(1,1.12491070647241e-268 / exp(normal_cdf_log(var(-35),0,1).val()));
  EXPECT_FLOAT_EQ(1,4.01072896657726e-261 / exp(normal_cdf_log(var(-34.5),0,1).val()));
  EXPECT_FLOAT_EQ(1,1.11389878557438e-253 / exp(normal_cdf_log(var(-34),0,1).val()));
  EXPECT_FLOAT_EQ(1,2.40983869512039e-246 / exp(normal_cdf_log(var(-33.5),0,1).val()));
  EXPECT_FLOAT_EQ(1,4.06118562091586e-239 / exp(normal_cdf_log(var(-33),0,1).val()));
  EXPECT_FLOAT_EQ(1,5.33142435967881e-232 / exp(normal_cdf_log(var(-32.5),0,1).val()));
  EXPECT_FLOAT_EQ(1,5.4520806035124e-225 / exp(normal_cdf_log(var(-32),0,1).val()));
  EXPECT_FLOAT_EQ(1,4.34323260103177e-218 / exp(normal_cdf_log(var(-31.5),0,1).val()));
  EXPECT_FLOAT_EQ(1,2.6952500812005e-211 / exp(normal_cdf_log(var(-31),0,1).val()));
  EXPECT_FLOAT_EQ(1,1.30293791317808e-204 / exp(normal_cdf_log(var(-30.5),0,1).val()));
  EXPECT_FLOAT_EQ(1,4.90671392714819e-198 / exp(normal_cdf_log(var(-30),0,1).val()));
  EXPECT_FLOAT_EQ(1,1.43947455222918e-191 / exp(normal_cdf_log(var(-29.5),0,1).val()));
  EXPECT_FLOAT_EQ(1,3.28978526670438e-185 / exp(normal_cdf_log(var(-29),0,1).val()));
  EXPECT_FLOAT_EQ(1,5.85714125380634e-179 / exp(normal_cdf_log(var(-28.5),0,1).val()));
  EXPECT_FLOAT_EQ(1,8.12386946965943e-173 / exp(normal_cdf_log(var(-28),0,1).val()));
  EXPECT_FLOAT_EQ(1,8.77817055687808e-167 / exp(normal_cdf_log(var(-27.5),0,1).val()));
  EXPECT_FLOAT_EQ(1,7.38948100688502e-161 / exp(normal_cdf_log(var(-27),0,1).val()));
  EXPECT_FLOAT_EQ(1,4.84616266030332e-155 / exp(normal_cdf_log(var(-26.5),0,1).val()));
  EXPECT_FLOAT_EQ(1,2.47606331550339e-149 / exp(normal_cdf_log(var(-26),0,1).val()));
  EXPECT_FLOAT_EQ(1,9.85623651896393e-144 / exp(normal_cdf_log(var(-25.5),0,1).val()));
  EXPECT_FLOAT_EQ(1,3.05669670638256e-138 / exp(normal_cdf_log(var(-25),0,1).val()));
  EXPECT_FLOAT_EQ(1,7.38570686148941e-133 / exp(normal_cdf_log(var(-24.5),0,1).val()));
  EXPECT_FLOAT_EQ(1,1.3903921185497e-127 / exp(normal_cdf_log(var(-24),0,1).val()));
  EXPECT_FLOAT_EQ(1,2.03936756324998e-122 / exp(normal_cdf_log(var(-23.5),0,1).val()));
  EXPECT_FLOAT_EQ(1,2.33063700622065e-117 / exp(normal_cdf_log(var(-23),0,1).val()));
  EXPECT_FLOAT_EQ(1,2.07531079906636e-112 / exp(normal_cdf_log(var(-22.5),0,1).val()));
  EXPECT_FLOAT_EQ(1,1.43989243514508e-107 / exp(normal_cdf_log(var(-22),0,1).val()));
  EXPECT_FLOAT_EQ(1,7.78439707718263e-103 / exp(normal_cdf_log(var(-21.5),0,1).val()));
  EXPECT_FLOAT_EQ(1,3.27927801897904e-98 / exp(normal_cdf_log(var(-21),0,1).val()));
  EXPECT_FLOAT_EQ(1,1.0764673258791e-93 / exp(normal_cdf_log(var(-20.5),0,1).val()));
  EXPECT_FLOAT_EQ(1,2.75362411860623e-89 / exp(normal_cdf_log(var(-20),0,1).val()));
  EXPECT_FLOAT_EQ(1,5.48911547566041e-85 / exp(normal_cdf_log(var(-19.5),0,1).val()));
  EXPECT_FLOAT_EQ(1,8.52722395263098e-81 / exp(normal_cdf_log(var(-19),0,1).val()));
  EXPECT_FLOAT_EQ(1,1.03236986895633e-76 / exp(normal_cdf_log(var(-18.5),0,1).val()));
  EXPECT_FLOAT_EQ(1,9.74094891893715e-73 / exp(normal_cdf_log(var(-18),0,1).val()));
  EXPECT_FLOAT_EQ(1,7.16345876623504e-69 / exp(normal_cdf_log(var(-17.5),0,1).val()));
  EXPECT_FLOAT_EQ(1,4.10599620209891e-65 / exp(normal_cdf_log(var(-17),0,1).val()));
  EXPECT_FLOAT_EQ(1,1.83446300316473e-61 / exp(normal_cdf_log(var(-16.5),0,1).val()));
  EXPECT_FLOAT_EQ(1,6.38875440053809e-58 / exp(normal_cdf_log(var(-16),0,1).val()));
  EXPECT_FLOAT_EQ(1, 1.73446079179387e-54 / exp(normal_cdf_log(var(-15.5),0,1).val()));
  EXPECT_FLOAT_EQ(1, 3.67096619931275e-51 / exp(normal_cdf_log(var(-15),0,1).val()));
  EXPECT_FLOAT_EQ(1, 6.05749476441522e-48 / exp(normal_cdf_log(var(-14.5),0,1).val()));
  EXPECT_FLOAT_EQ(1, 7.7935368191928e-45 / exp(normal_cdf_log(var(-14),0,1).val()));
  EXPECT_FLOAT_EQ(1, 7.81880730565789e-42 / exp(normal_cdf_log(var(-13.5),0,1).val()));
  EXPECT_FLOAT_EQ(1, 6.11716439954988e-39 / exp(normal_cdf_log(var(-13),0,1).val()));
  EXPECT_FLOAT_EQ(1, 3.73256429887771e-36 / exp(normal_cdf_log(var(-12.5),0,1).val()));
  EXPECT_FLOAT_EQ(1, 1.77648211207768e-33 / exp(normal_cdf_log(var(-12),0,1).val()));
 EXPECT_FLOAT_EQ(1, 6.59577144611367e-31 / exp(normal_cdf_log(var(-11.5),0,1).val()));
  EXPECT_FLOAT_EQ(1, 1.91065957449868e-28 / exp(normal_cdf_log(var(-11),0,1).val()));
  EXPECT_FLOAT_EQ(1, 4.31900631780923e-26 / exp(normal_cdf_log(var(-10.5),0,1).val()));
  EXPECT_FLOAT_EQ(1, 7.61985302416053e-24 / exp(normal_cdf_log(var(-10),0,1).val()));
  EXPECT_FLOAT_EQ(1, 1.04945150753626e-21 / exp(normal_cdf_log(var(-9.5),0,1).val()));
  EXPECT_FLOAT_EQ(1, 1.12858840595384e-19 / exp(normal_cdf_log(var(-9),0,1).val()));
  EXPECT_FLOAT_EQ(1, 9.47953482220332e-18 / exp(normal_cdf_log(var(-8.5),0,1).val()));
  EXPECT_FLOAT_EQ(1, 6.22096057427178e-16 / exp(normal_cdf_log(var(-8),0,1).val()));
  EXPECT_FLOAT_EQ(1, 3.1908916729109e-14 / exp(normal_cdf_log(var(-7.5),0,1).val()));
  EXPECT_FLOAT_EQ(1, 1.27981254388584e-12 / exp(normal_cdf_log(var(-7),0,1).val()));
  EXPECT_FLOAT_EQ(1, 4.01600058385912e-11 / exp(normal_cdf_log(var(-6.5),0,1).val()));
  EXPECT_FLOAT_EQ(1, 9.86587645037698e-10 / exp(normal_cdf_log(var(-6),0,1).val()));
  EXPECT_FLOAT_EQ(1, 1.89895624658877e-08 / exp(normal_cdf_log(var(-5.5),0,1).val()));
  EXPECT_FLOAT_EQ(1, 2.86651571879194e-07 / exp(normal_cdf_log(var(-5),0,1).val()));
  EXPECT_FLOAT_EQ(1, 3.39767312473006e-06 / exp(normal_cdf_log(var(-4.5),0,1).val()));
  EXPECT_FLOAT_EQ(1, 3.16712418331199e-05 / exp(normal_cdf_log(var(-4),0,1).val()));
  EXPECT_FLOAT_EQ(1, 0.000232629079035525 / exp(normal_cdf_log(var(-3.5),0,1).val()));
  EXPECT_FLOAT_EQ(1, 0.00134989803163009 / exp(normal_cdf_log(var(-3),0,1).val()));
  EXPECT_FLOAT_EQ(1, 0.00620966532577613 / exp(normal_cdf_log(var(-2.5),0,1).val()));
  EXPECT_FLOAT_EQ(1, 0.0227501319481792 / exp(normal_cdf_log(var(-2),0,1).val()));
  EXPECT_FLOAT_EQ(1, 0.0668072012688581 / exp(normal_cdf_log(var(-1.5),0,1).val()));
  EXPECT_FLOAT_EQ(1, 0.158655253931457 / exp(normal_cdf_log(var(-1),0,1).val()));
  EXPECT_FLOAT_EQ(1, 0.308537538725987 / exp(normal_cdf_log(var(-0.5),0,1).val()));
  EXPECT_FLOAT_EQ(1, 0.5 / exp(normal_cdf_log(var(0),0,1).val()));
  EXPECT_FLOAT_EQ(1, 0.691462461274013 / exp(normal_cdf_log(var(0.5),0,1).val()));
  EXPECT_FLOAT_EQ(1, 0.841344746068543 / exp(normal_cdf_log(var(1),0,1).val()));
  EXPECT_FLOAT_EQ(1, 0.933192798731142 / exp(normal_cdf_log(var(1.5),0,1).val()));
  EXPECT_FLOAT_EQ(1, 0.977249868051821 / exp(normal_cdf_log(var(2),0,1).val()));
  EXPECT_FLOAT_EQ(1, 0.993790334674224 / exp(normal_cdf_log(var(2.5),0,1).val()));
  EXPECT_FLOAT_EQ(1, 0.99865010196837 / exp(normal_cdf_log(var(3),0,1).val()));
  EXPECT_FLOAT_EQ(1, 0.999767370920964 / exp(normal_cdf_log(var(3.5),0,1).val()));
  EXPECT_FLOAT_EQ(1, 0.999968328758167 / exp(normal_cdf_log(var(4),0,1).val()));
  EXPECT_FLOAT_EQ(1, 0.999996602326875 / exp(normal_cdf_log(var(4.5),0,1).val()));
  EXPECT_FLOAT_EQ(1, 0.999999713348428 / exp(normal_cdf_log(var(5),0,1).val()));
  EXPECT_FLOAT_EQ(1, 0.999999981010438 / exp(normal_cdf_log(var(5.5),0,1).val()));
  EXPECT_FLOAT_EQ(1, 0.999999999013412 / exp(normal_cdf_log(var(6),0,1).val()));
  EXPECT_FLOAT_EQ(1, 0.99999999995984 / exp(normal_cdf_log(var(6.5),0,1).val()));
  EXPECT_FLOAT_EQ(1, 0.99999999999872 / exp(normal_cdf_log(var(7),0,1).val()));
  EXPECT_FLOAT_EQ(1, 0.999999999999968 / exp(normal_cdf_log(var(7.5),0,1).val()));
  EXPECT_FLOAT_EQ(1, 0.999999999999999 / exp(normal_cdf_log(var(8),0,1).val()));
  EXPECT_FLOAT_EQ(1, 1 / exp(normal_cdf_log(var(8.5),0,1).val()));
  EXPECT_FLOAT_EQ(1, 1 / exp(normal_cdf_log(var(9),0,1).val()));
  EXPECT_FLOAT_EQ(1, 1 / exp(normal_cdf_log(var(9.5),0,1).val()));
  EXPECT_FLOAT_EQ(1, 1 / exp(normal_cdf_log(var(10),0,1).val())); 
}

TEST(ProbDistributionsNormal, ccdf_log_tail) {
   using stan::agrad::var;
   using stan::prob::normal_ccdf_log;
   using std::exp;

   EXPECT_FLOAT_EQ(1, -6.661338147750941214694e-16/(normal_ccdf_log(var(-8.0),0,1).val()));
   EXPECT_FLOAT_EQ(1, -3.186340080674249758114e-14/(normal_ccdf_log(var(-7.5),0,1).val()));
   EXPECT_FLOAT_EQ(1, -1.279865102788699562477e-12/(normal_ccdf_log(var(-7.0),0,1).val()));
   EXPECT_FLOAT_EQ(1, -4.015998644826973564545e-11/(normal_ccdf_log(var(-6.5),0,1).val()));

   EXPECT_FLOAT_EQ(1, -9.865877009111571184118e-10/(normal_ccdf_log(var(-6.0),0,1).val()));
   EXPECT_FLOAT_EQ(1, -1.898956265833514866414e-08/(normal_ccdf_log(var(-5.5),0,1).val()));
   EXPECT_FLOAT_EQ(1, -2.866516130081049047962e-07/(normal_ccdf_log(var(-5.0),0,1).val()));
   EXPECT_FLOAT_EQ(1, -3.397678896843115195074e-06/(normal_ccdf_log(var(-4.5),0,1).val()));
   EXPECT_FLOAT_EQ(1, -3.167174337748932124543e-05/(normal_ccdf_log(var(-4.0),0,1).val()));
   EXPECT_FLOAT_EQ(1, -0.0002326561413768195969113/(normal_ccdf_log(var(-3.5),0,1).val()));
   EXPECT_FLOAT_EQ(1, -0.001350809964748202673598/(normal_ccdf_log(var(-3.0),0,1).val()));
   EXPECT_FLOAT_EQ(1, -0.0062290254858600267035/(normal_ccdf_log(var(-2.5),0,1).val()));
   EXPECT_FLOAT_EQ(1, -0.02301290932896348992442/(normal_ccdf_log(var(-2.0),0,1).val()));
   EXPECT_FLOAT_EQ(1, -0.06914345561223400604689/(normal_ccdf_log(var(-1.5),0,1).val()));
   EXPECT_FLOAT_EQ(1, -0.1727537790234499048836/(normal_ccdf_log(var(-1.0),0,1).val()));
   EXPECT_FLOAT_EQ(1, -0.3689464152886565151412/(normal_ccdf_log(var(-0.5),0,1).val()));
   EXPECT_FLOAT_EQ(1, -0.6931471805599452862268/(normal_ccdf_log(var(0),0,1).val()));
   EXPECT_FLOAT_EQ(1, -1.175911761593618320987/(normal_ccdf_log(var(0.5),0,1).val()));
   EXPECT_FLOAT_EQ(1, -1.841021645009263352222/(normal_ccdf_log(var(1.0),0,1).val()));
   EXPECT_FLOAT_EQ(1, -2.705944400823889317564/(normal_ccdf_log(var(1.5),0,1).val()));
   EXPECT_FLOAT_EQ(1, -3.78318433368203210776/(normal_ccdf_log(var(2.0),0,1).val()));
   EXPECT_FLOAT_EQ(1, -5.081648277278686620662/(normal_ccdf_log(var(2.5),0,1).val()));
   EXPECT_FLOAT_EQ(1, -6.607726221510342945464/(normal_ccdf_log(var(3.0),0,1).val()));
   EXPECT_FLOAT_EQ(1, -8.366065308344028395027/(normal_ccdf_log(var(3.5),0,1).val()));
   EXPECT_FLOAT_EQ(1, -10.36010148652728979357/(normal_ccdf_log(var(4.0),0,1).val()));
   EXPECT_FLOAT_EQ(1, -12.59241973571053385683/(normal_ccdf_log(var(4.5),0,1).val()));
   EXPECT_FLOAT_EQ(1, -15.06499839383403838156/(normal_ccdf_log(var(5.0),0,1).val()));
   EXPECT_FLOAT_EQ(1, -17.77937635198566113104/(normal_ccdf_log(var(5.5),0,1).val()));
   EXPECT_FLOAT_EQ(1, -20.73676889383495947072/(normal_ccdf_log(var(6.0),0,1).val()));
   EXPECT_FLOAT_EQ(1, -23.93814997800869548428/(normal_ccdf_log(var(6.5),0,1).val()));
}


TEST(ProbDistributionsNormal, intVsDouble) {
  using stan::agrad::var;
  for (double thetaval = -5.0; thetaval < 6.0; thetaval += 0.5) {
    var theta(thetaval);
    var lp1(0.0);
    lp1 += stan::prob::normal_log<true>(0, theta, 1);
    double lp1val = lp1.val();
    stan::agrad::grad(lp1.vi_); 
    double lp1adj = lp1.adj();

    var theta2(thetaval);
    var lp2(0.0);
    lp2 += stan::prob::normal_log<true>(theta2, 0, 1);
    double lp2val = lp2.val();
    stan::agrad::grad(lp2.vi_);
    double lp2adj = lp2.adj();
    EXPECT_FLOAT_EQ(lp1val,lp2val);
    EXPECT_FLOAT_EQ(lp1adj,lp2adj);
  }
}

TEST(ProbDistributionsNormal, error_check) {
  boost::random::mt19937 rng;
  EXPECT_NO_THROW(stan::prob::normal_rng(10.0,2.0,rng));

  EXPECT_THROW(stan::prob::normal_rng(10.0,-2.0,rng),std::domain_error);
  EXPECT_THROW(stan::prob::normal_rng(10.0,0,rng),std::domain_error);
  EXPECT_THROW(stan::prob::normal_rng(stan::math::positive_infinity(),-2.0,rng),
               std::domain_error);
  EXPECT_THROW(stan::prob::normal_rng(2,stan::math::negative_infinity(),rng),
               std::domain_error);
}

TEST(ProbDistributionsNormal, chiSquareGoodnessFitTest) {
  boost::random::mt19937 rng;
  int N = 10000;
  int K = boost::math::round(2 * std::pow(N, 0.4));
  boost::math::normal_distribution<>dist (2.0,1.0);
  boost::math::chi_squared mydist(K-1);

  double loc[K - 1];
  for(int i = 1; i < K; i++)
    loc[i - 1] = quantile(dist, i * std::pow(K, -1.0));

  int count = 0;
  int bin [K];
  double expect [K];
  for(int i = 0 ; i < K; i++) {
    bin[i] = 0;
    expect[i] = N / K;
  }

  while (count < N) {
    double a = stan::prob::normal_rng(2.0,1.0,rng);
    int i = 0;
    while (i < K-1 && a > loc[i]) 
      ++i;
    ++bin[i];
    count++;
   }

  double chi = 0;

  for(int j = 0; j < K; j++)
    chi += ((bin[j] - expect[j]) * (bin[j] - expect[j]) / expect[j]);

  EXPECT_TRUE(chi < quantile(complement(mydist, 1e-6)));
}

