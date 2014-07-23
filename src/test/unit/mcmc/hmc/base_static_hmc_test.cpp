#include <test/unit/mcmc/hmc/mock_hmc.hpp>
#include <stan/mcmc/hmc/static/base_static_hmc.hpp>

#include <boost/random/additive_combine.hpp>

#include <gtest/gtest.h>

typedef boost::ecuyer1988 rng_t;


namespace stan {
  
  namespace mcmc {
    
    // Mock Static HMC
    class mock_static_hmc: public base_static_hmc<mock_model,
                                                  ps_point,
                                                  mock_hamiltonian,
                                                  mock_integrator,
                                                  rng_t> {
      
    public:
      
      mock_static_hmc(mock_model &m, rng_t& rng, std::ostream* o, std::ostream* e)
        : base_static_hmc<mock_model,ps_point,mock_hamiltonian,mock_integrator,rng_t>(m, rng, o, e)
      { this->name_ = "Mock Static HMC"; }
      
    };
    
  }
  
}

TEST(McmcBaseStaticHMC, set_nominal_stepsize) {
  
  rng_t base_rng(0);
  
  std::vector<double> q(5, 1.0);
  std::vector<int> r(2, 2);
  
  stan::mcmc::mock_model model(q.size());

  std::stringstream output, error;
  
  stan::mcmc::mock_static_hmc sampler(model, base_rng, &output, &error);
  
  double old_epsilon = 1.0;
  
  sampler.set_nominal_stepsize(old_epsilon);
  EXPECT_EQ(old_epsilon, sampler.get_nominal_stepsize());
  EXPECT_EQ(true, sampler.get_L() > 0);
  
  sampler.set_nominal_stepsize(-0.1);
  EXPECT_EQ(old_epsilon, sampler.get_nominal_stepsize());
  
  EXPECT_EQ("", output.str());
  EXPECT_EQ("", error.str());
}

TEST(McmcBaseStaticHMC, set_T) {
  
  rng_t base_rng(0);
  
  std::vector<double> q(5, 1.0);
  std::vector<int> r(2, 2);
  
  stan::mcmc::mock_model model(q.size());
  
  std::stringstream output, error;
  stan::mcmc::mock_static_hmc sampler(model, base_rng, &output, &error);
  
  double old_T = 3.0;
  
  sampler.set_T(old_T);
  EXPECT_EQ(old_T, sampler.get_T());
  EXPECT_EQ(true, sampler.get_L() > 0);
  
  sampler.set_T(-0.1);
  EXPECT_EQ(old_T, sampler.get_T());
  
  EXPECT_EQ("", output.str());
  EXPECT_EQ("", error.str());
}

TEST(McmcBaseStaticHMC, set_nominal_stepsize_and_T) {
  
  rng_t base_rng(0);
  
  std::vector<double> q(5, 1.0);
  std::vector<int> r(2, 2);
  
  stan::mcmc::mock_model model(q.size());
  
  std::stringstream output, error;
  stan::mcmc::mock_static_hmc sampler(model, base_rng, &output, &error);
  
  double old_epsilon = 1.0;
  double old_T = 3.0;
  
  sampler.set_nominal_stepsize_and_T(old_epsilon, old_T);
  EXPECT_EQ(old_epsilon, sampler.get_nominal_stepsize());
  EXPECT_EQ(old_T, sampler.get_T());
  EXPECT_EQ(true, sampler.get_L() > 0);
  
  sampler.set_nominal_stepsize_and_T(-0.1, 5.0);
  EXPECT_EQ(old_epsilon, sampler.get_nominal_stepsize());
  EXPECT_EQ(old_T, sampler.get_T());
  
  sampler.set_nominal_stepsize_and_T(5.0, -0.1);
  EXPECT_EQ(old_epsilon, sampler.get_nominal_stepsize());
  EXPECT_EQ(old_T, sampler.get_T());
  
  EXPECT_EQ("", output.str());
  EXPECT_EQ("", error.str());
}

TEST(McmcBaseStaticHMC, set_nominal_stepsize_and_L) {
  
  rng_t base_rng(0);
  
  std::vector<double> q(5, 1.0);
  std::vector<int> r(2, 2);
  
  stan::mcmc::mock_model model(q.size());
  
  std::stringstream output, error;
  stan::mcmc::mock_static_hmc sampler(model, base_rng, &output, &error);
  
  double old_epsilon = 1.0;
  int old_L = 10;
  
  sampler.set_nominal_stepsize_and_L(old_epsilon, old_L);
  EXPECT_EQ(old_epsilon, sampler.get_nominal_stepsize());
  EXPECT_EQ(old_L, sampler.get_L());
  EXPECT_EQ(true, sampler.get_T() > 0);
  
  sampler.set_nominal_stepsize_and_L(-0.1, 5);
  EXPECT_EQ(old_epsilon, sampler.get_nominal_stepsize());
  EXPECT_EQ(old_L, sampler.get_L());
  
  sampler.set_nominal_stepsize_and_T(5.0, -1);
  EXPECT_EQ(old_epsilon, sampler.get_nominal_stepsize());
  EXPECT_EQ(old_L, sampler.get_L());
  
  EXPECT_EQ("", output.str());
  EXPECT_EQ("", error.str());
}
