#include <test/mcmc/mock_hmc.hpp>
#include <stan/mcmc/hmc/hamiltonians/diag_e_metric.hpp>

#include <boost/random/additive_combine.hpp>

#include <gtest/gtest.h>

typedef boost::ecuyer1988 rng_t;

TEST(McmcDiagEMetric, sample_p) {
  
  rng_t base_rng(0);
  
  std::vector<double> q(5, 1.0);
  std::vector<int> r(2, 2);
  
  stan::mcmc::mock_model model(q.size());
  
  stan::mcmc::diag_e_metric<stan::mcmc::mock_model, rng_t> metric(model,&std::cout);
  stan::mcmc::diag_e_point z(q.size(), r.size());
  
  int n_samples = 1000;
  double m = 0;
  double m2 = 0;
  
  for (int i = 0; i < n_samples; ++i) {
    metric.sample_p(z, base_rng);
    double T = metric.T(z);
    
    double delta = T - m;
    m += delta / static_cast<double>(i + 1);
    m2 += delta * (T - m);
  }
  
  double var = m2 / (n_samples + 1.0);
  
  // Mean within 5sigma of expected value (d / 2)
  EXPECT_EQ(true, fabs(m   - 0.5 * q.size()) < 5.0 * sqrt(var));
  
  // Variance within 10% of expected value (d / 2)
  EXPECT_EQ(true, fabs(var - 0.5 * q.size()) < 0.1 * q.size());
  
}