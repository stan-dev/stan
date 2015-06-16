#include <stan/math/prim/scal/prob/normal_ccdf_log.hpp>
#include <gtest/gtest.h>
#include <stan/math/rev/scal/fun/value_of.hpp>
#include <stan/math/rev/scal/fun/value_of_rec.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/prim/scal/fun/is_nan.hpp>

TEST(normal_ccdf_log, tail) {
   using stan::math::var;
   using stan::math::normal_ccdf_log;
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

  stan::math::recover_memory();
}

void test_value_and_derivatives(double expected_val,
                                double y_dbl, double mu_dbl, double sigma_dbl) {
  using stan::math::is_nan;
  using stan::math::var;  
  using stan::math::normal_ccdf_log;
  std::stringstream msg_ss;
  msg_ss << "parameters: (" << y_dbl << ", " << mu_dbl << ", " << sigma_dbl << ")";
  std::string msg = msg_ss.str();

  SCOPED_TRACE(msg);
  
  var y(y_dbl);
  var mu(mu_dbl);
  var sigma(sigma_dbl);

  std::vector<double> gradients;
  var val = normal_ccdf_log(y, mu, sigma);
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
  finite_diffs[0] = (normal_ccdf_log(y_dbl + e, mu_dbl, sigma_dbl)
                     - normal_ccdf_log(y_dbl - e, mu_dbl, sigma_dbl)) * inv2e;
  finite_diffs[1] = (normal_ccdf_log(y_dbl, mu_dbl + e, sigma_dbl)
                     - normal_ccdf_log(y_dbl, mu_dbl - e, sigma_dbl)) * inv2e;
  finite_diffs[2] = (normal_ccdf_log(y_dbl, mu_dbl, sigma_dbl + e)
                     - normal_ccdf_log(y_dbl, mu_dbl, sigma_dbl - e)) * inv2e;

  
  EXPECT_FLOAT_EQ(expected_val, val.val());
  EXPECT_FALSE(is_nan(gradients[0]));
  EXPECT_FALSE(is_nan(gradients[1]));
  EXPECT_FALSE(is_nan(gradients[2]));
  if (!is_nan(gradients[0])) {
    if (!is_nan(finite_diffs[0]))
      EXPECT_NEAR(finite_diffs[0], gradients[0], 1e-2);
    else
      EXPECT_FLOAT_EQ(-std::numeric_limits<double>::infinity(), gradients[0]);
  }
  if (!is_nan(gradients[1])) {
    if (!is_nan(finite_diffs[1]))
      EXPECT_NEAR(finite_diffs[1], gradients[1], 1e-2);
    else
      EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), gradients[1]);
  }
  if (!is_nan(gradients[2])) {
    if (!is_nan(finite_diffs[2]))
      EXPECT_NEAR(finite_diffs[2], gradients[2], 1e-2);
    else
      EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), gradients[2]);
  }
}


TEST(normal_ccdf_log, derivatives) {
  test_value_and_derivatives(log(0.5), 10.0, 10.0, 0.5);
  test_value_and_derivatives(0.0, -20.0, 10.0, 0.5);
  test_value_and_derivatives(0.0, -30.0, 10.0, 0.5);
  test_value_and_derivatives(0.0, -50.0, 10.0, 1.0);

  test_value_and_derivatives(-std::numeric_limits<double>::infinity(), 30.0, 10.0, 0.5);
}
