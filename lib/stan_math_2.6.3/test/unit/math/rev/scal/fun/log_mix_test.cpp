#include <vector>
#include <stan/math/rev/scal/fun/log_mix.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <test/unit/math/rev/scal/fun/nan_util.hpp>
#include <gtest/gtest.h>
#include <stan/math/rev/core.hpp>
#include <stan/math/rev/scal/fun/exp.hpp>
#include <stan/math/rev/scal/fun/log.hpp>

void test_log_mix_vvv(double theta, double lambda1, double lambda2) {
  using stan::math::var;

  var theta_v = theta;
  var lambda1_v = lambda1;
  var lambda2_v = lambda2;

  std::vector<var> x;
  x.push_back(theta_v);
  x.push_back(lambda1_v);
  x.push_back(lambda2_v);

  var lp = log_mix(theta_v,lambda1_v,lambda2_v);
  double val1 = lp.val();

  std::vector<double> g;
  lp.grad(x,g);
  
  stan::math::recover_memory();
  
  var theta_v2 = theta;
  var lambda1_v2 = lambda1;
  var lambda2_v2 = lambda2;
  
  std::vector<var> x2;
  x2.push_back(theta_v2);
  x2.push_back(lambda1_v2);
  x2.push_back(lambda2_v2);

  var lp2 = log(theta_v2 * exp(lambda1_v2) + (1 - theta_v2) * exp(lambda2_v2));
  double val2 = lp2.val();

  std::vector<double> g2;
  lp2.grad(x2,g2);

  EXPECT_FLOAT_EQ(val2, val1);
  EXPECT_EQ(g2.size(), g.size());
  for (size_t i = 0; i < g2.size(); ++i)
    EXPECT_FLOAT_EQ(g2[i], g[i]) << "failed on " << i << std::endl;

  var theta_v3 = 2.0;
  EXPECT_THROW(log_mix(theta_v3,lambda1_v,lambda2_v),std::domain_error);
  stan::math::recover_memory();
}


void test_log_mix_vv_ex_lam_2(double theta, double lambda1, double lambda2) {
  using stan::math::var;

  var theta_v = theta;
  var lambda1_v = lambda1;

  std::vector<var> x;
  x.push_back(theta_v);
  x.push_back(lambda1_v);

  var lp = log_mix(theta_v,lambda1_v,lambda2);
  double val1 = lp.val();

  std::vector<double> g;
  lp.grad(x,g);
  
  stan::math::recover_memory();
  
  var theta_v2 = theta;
  var lambda1_v2 = lambda1;
  
  std::vector<var> x2;
  x2.push_back(theta_v2);
  x2.push_back(lambda1_v2);

  var lp2 = log(theta_v2 * exp(lambda1_v2) + (1 - theta_v2) * exp(lambda2));
  double val2 = lp2.val();

  std::vector<double> g2;
  lp2.grad(x2,g2);

  EXPECT_FLOAT_EQ(val2, val1);
  EXPECT_EQ(g2.size(), g.size());
  for (size_t i = 0; i < g2.size(); ++i)
    EXPECT_FLOAT_EQ(g2[i], g[i]) << "failed on " << i << std::endl;

  var theta_v3 = 2.0;
  EXPECT_THROW(log_mix(theta_v3,lambda1_v,lambda2),std::domain_error);
  stan::math::recover_memory();
}

void test_log_mix_vv_ex_lam_1(double theta, double lambda1, double lambda2) {
  using stan::math::var;

  var theta_v = theta;
  var lambda2_v = lambda2;

  std::vector<var> x;
  x.push_back(theta_v);
  x.push_back(lambda2_v);

  var lp = log_mix(theta_v,lambda1,lambda2_v);
  double val1 = lp.val();

  std::vector<double> g;
  lp.grad(x,g);
  
  stan::math::recover_memory();
  
  var theta_v2 = theta;
  var lambda2_v2 = lambda2;
  
  std::vector<var> x2;
  x2.push_back(theta_v2);
  x2.push_back(lambda2_v2);

  var lp2 = log(theta_v2 * exp(lambda1) + (1 - theta_v2) * exp(lambda2_v2));
  double val2 = lp2.val();

  std::vector<double> g2;
  lp2.grad(x2,g2);

  EXPECT_FLOAT_EQ(val2, val1);
  EXPECT_EQ(g2.size(), g.size());
  for (size_t i = 0; i < g2.size(); ++i)
    EXPECT_FLOAT_EQ(g2[i], g[i]) << "failed on " << i << std::endl;

  var theta_v3 = 2.0;
  EXPECT_THROW(log_mix(theta_v3,lambda1,lambda2_v),std::domain_error);
  stan::math::recover_memory();
}

void test_log_mix_vv_ex_theta(double theta, double lambda1, double lambda2) {
  using stan::math::var;

  var lambda1_v = lambda1;
  var lambda2_v = lambda2;

  std::vector<var> x;
  x.push_back(lambda1_v);
  x.push_back(lambda2_v);

  var lp = log_mix(theta,lambda1_v,lambda2_v);
  double val1 = lp.val();

  std::vector<double> g;
  lp.grad(x,g);
  
  stan::math::recover_memory();
  
  var lambda1_v2 = lambda1;
  var lambda2_v2 = lambda2;
  
  std::vector<var> x2;
  x2.push_back(lambda1_v2);
  x2.push_back(lambda2_v2);

  var lp2 = log(theta * exp(lambda1_v2) + (1 - theta) * exp(lambda2_v2));
  double val2 = lp2.val();

  std::vector<double> g2;
  lp2.grad(x2,g2);

  EXPECT_FLOAT_EQ(val2, val1);
  EXPECT_EQ(g2.size(), g.size());
  for (size_t i = 0; i < g2.size(); ++i)
    EXPECT_FLOAT_EQ(g2[i], g[i]) << "failed on " << i << std::endl;

  EXPECT_THROW(log_mix(2.0,lambda1_v,lambda2_v),std::domain_error);
  stan::math::recover_memory();
}

void test_log_mix_v_theta(double theta, double lambda1, double lambda2) {
  using stan::math::var;

  var theta_v = theta;

  std::vector<var> x;
  x.push_back(theta_v);

  var lp = log_mix(theta_v,lambda1,lambda2);
  double val1 = lp.val();

  std::vector<double> g;
  lp.grad(x,g);
  
  stan::math::recover_memory();
  
  var theta_v2 = theta;
  
  std::vector<var> x2;
  x2.push_back(theta_v2);

  var lp2 = log(theta_v2 * exp(lambda1) + (1 - theta_v2) * exp(lambda2));
  double val2 = lp2.val();

  std::vector<double> g2;
  lp2.grad(x2,g2);

  EXPECT_FLOAT_EQ(val2, val1);
  EXPECT_EQ(g2.size(), g.size());
  for (size_t i = 0; i < g2.size(); ++i)
    EXPECT_FLOAT_EQ(g2[i], g[i]) << "failed on " << i << std::endl;

  var theta_v3 = 2.0;
  EXPECT_THROW(log_mix(theta_v3,lambda1,lambda2),std::domain_error);
  stan::math::recover_memory();
}

void test_log_mix_v_lam_1(double theta, double lambda1, double lambda2) {
  using stan::math::var;

  var lambda1_v = lambda1;

  std::vector<var> x;
  x.push_back(lambda1_v);

  var lp = log_mix(theta,lambda1_v,lambda2);
  double val1 = lp.val();

  std::vector<double> g;
  lp.grad(x,g);
  
  stan::math::recover_memory();
  
  var lambda1_v2 = lambda1;
  
  std::vector<var> x2;
  x2.push_back(lambda1_v2);

  var lp2 = log(theta * exp(lambda1_v2) + (1 - theta) * exp(lambda2));
  double val2 = lp2.val();

  std::vector<double> g2;
  lp2.grad(x2,g2);

  EXPECT_FLOAT_EQ(val2, val1);
  EXPECT_EQ(g2.size(), g.size());
  for (size_t i = 0; i < g2.size(); ++i)
    EXPECT_FLOAT_EQ(g2[i], g[i]) << "failed on " << i << std::endl;

  EXPECT_THROW(log_mix(2.0,lambda1_v,lambda2),std::domain_error);
  stan::math::recover_memory();
}

void test_log_mix_v_lam_2(double theta, double lambda1, double lambda2) {
  using stan::math::var;

  var lambda2_v = lambda2;

  std::vector<var> x;
  x.push_back(lambda2_v);

  var lp = log_mix(theta,lambda1,lambda2_v);
  double val1 = lp.val();

  std::vector<double> g;
  lp.grad(x,g);
  
  stan::math::recover_memory();
  
  var lambda2_v2 = lambda2;
  
  std::vector<var> x2;
  x2.push_back(lambda2_v2);

  var lp2 = log(theta * exp(lambda1) + (1 - theta) * exp(lambda2_v2));
  double val2 = lp2.val();

  std::vector<double> g2;
  lp2.grad(x2,g2);

  EXPECT_FLOAT_EQ(val2, val1);
  EXPECT_EQ(g2.size(), g.size());
  for (size_t i = 0; i < g2.size(); ++i)
    EXPECT_FLOAT_EQ(g2[i], g[i]) << "failed on " << i << std::endl;

  EXPECT_THROW(log_mix(2.0,lambda1,lambda2_v),std::domain_error);
  stan::math::recover_memory();
}

TEST(AgradRev,log_mix) {

  test_log_mix_vvv(0.3, 10.4, -10.9);
  test_log_mix_vvv(0.7, -4.7, 10.1);

  test_log_mix_vv_ex_lam_2(0.7, 1.4, -1.9);
  test_log_mix_vv_ex_lam_2(0.3, -10.4, 7.8);

  test_log_mix_vv_ex_lam_1(0.7, 1.4, -1.9);
  test_log_mix_vv_ex_lam_1(0.3, -10.4, 7.8);

  test_log_mix_vv_ex_theta(0.7, 1.4, -1.9);
  test_log_mix_vv_ex_theta(0.3, -10.4, 7.8);

  test_log_mix_v_theta(0.7, 1.4, -1.9);
  test_log_mix_v_theta(0.3, -1.4, 1.7);

  test_log_mix_v_lam_1(0.7, 1.4, -1.9);
  test_log_mix_v_lam_1(0.8, -1.9, 10.9);

  test_log_mix_v_lam_2(0.7, 1.4, 3.99);
  test_log_mix_v_lam_2(0.1, -1.4, 3.99);
}

struct log_mix_fun {
  template <typename T0, typename T1, typename T2>
  inline
  typename stan::return_type<T0,T1,T2>::type
  operator()(const T0& arg1,
             const T1& arg2,
             const T2& arg3) const {
    return log_mix(arg1,arg2,arg3);
  }
};

TEST(AgradRev,log_mix_NaN) {
  log_mix_fun log_mix_;
  test_nan(log_mix_,0.6,0.3,0.5,true,false);
}
