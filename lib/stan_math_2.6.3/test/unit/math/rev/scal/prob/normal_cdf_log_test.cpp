#include <stan/math/prim/scal/prob/normal_cdf_log.hpp>
#include <gtest/gtest.h>
#include <stan/math/rev/scal/fun/value_of.hpp>
#include <stan/math/rev/scal/fun/value_of_rec.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/prim/scal/fun/is_nan.hpp>

TEST(normal_cdf_log, tails) {
  using stan::math::var;
  using stan::math::normal_cdf_log;
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

  stan::math::recover_memory();
}

void test_value_and_derivatives(double expected_val,
                                double y_dbl, double mu_dbl, double sigma_dbl) {
  using stan::math::is_nan;
  using stan::math::var;  
  using stan::math::normal_cdf_log;
  std::stringstream msg_ss;
  msg_ss << "parameters: (" << y_dbl << ", " << mu_dbl << ", " << sigma_dbl << ")";
  std::string msg = msg_ss.str();

  SCOPED_TRACE(msg);
  
  var y(y_dbl);
  var mu(mu_dbl);
  var sigma(sigma_dbl);

  std::vector<double> gradients;
  var val = normal_cdf_log(y, mu, sigma);
  std::vector<var> x;
  x.push_back(y);
  x.push_back(mu);
  x.push_back(sigma);
  gradients.clear();
  val.grad(x, gradients);

  double e = 1e-10;
  double inv2e = 0.5 / e;
  std::vector<double> finite_diffs;
  finite_diffs.resize(3);
  finite_diffs[0] = (normal_cdf_log(y_dbl + e, mu_dbl, sigma_dbl)
                     - normal_cdf_log(y_dbl - e, mu_dbl, sigma_dbl)) * inv2e;
  finite_diffs[1] = (normal_cdf_log(y_dbl, mu_dbl + e, sigma_dbl)
                     - normal_cdf_log(y_dbl, mu_dbl - e, sigma_dbl)) * inv2e;
  finite_diffs[2] = (normal_cdf_log(y_dbl, mu_dbl, sigma_dbl + e)
                     - normal_cdf_log(y_dbl, mu_dbl, sigma_dbl - e)) * inv2e;

  
  EXPECT_FLOAT_EQ(expected_val, val.val());
  EXPECT_FALSE(is_nan(gradients[0]));
  EXPECT_FALSE(is_nan(gradients[1]));
  EXPECT_FALSE(is_nan(gradients[2]));
  if (!is_nan(gradients[0])) {
    if (!is_nan(finite_diffs[0]))
      EXPECT_NEAR(finite_diffs[0], gradients[0], 1e-2);
    else
      EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), gradients[0]);
  }
  if (!is_nan(gradients[1])) {
    if (!is_nan(finite_diffs[1]))
      EXPECT_NEAR(finite_diffs[1], gradients[1], 1e-2);
    else
      EXPECT_FLOAT_EQ(-std::numeric_limits<double>::infinity(), gradients[1]);
  }
  if (!is_nan(gradients[2])) {
    if (!is_nan(finite_diffs[2]))
      EXPECT_NEAR(finite_diffs[2], gradients[2], 1e-2);
    else
      EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), gradients[2]);
  }
}


TEST(normal_cdf_log, derivatives) {
  test_value_and_derivatives(log(0.5), 10.0, 10.0, 0.5);
  test_value_and_derivatives(-std::numeric_limits<double>::infinity(), -20.0, 10.0, 0.5);
  test_value_and_derivatives(-std::numeric_limits<double>::infinity(), -30.0, 10.0, 0.5);
  test_value_and_derivatives(-std::numeric_limits<double>::infinity(), -50.0, 10.0, 1.0);

  test_value_and_derivatives(0.0, 30.0, 10.0, 0.5);
}
