#include <vector>
#include <stan/agrad/rev/functions/log_mix.hpp>
#include <test/unit/agrad/util.hpp>
#include <gtest/gtest.h>


void test_log_mix_vvv(double theta, double lambda1, double lambda2) {
  using stan::agrad::var;

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
  
  stan::agrad::recover_memory();
  
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
  stan::agrad::recover_memory();
}

TEST(AgradRev,log_mix) {

  test_log_mix_vvv(0.3, 10.4, -10.9);
}
