#include <test/unit/mcmc/hmc/mock_hmc.hpp>
#include <stan/mcmc/hmc/base_hmc.hpp>

#include <boost/random/additive_combine.hpp>
#include <boost/algorithm/string/split.hpp>

#include <gtest/gtest.h>

typedef boost::ecuyer1988 rng_t;


namespace stan {
  
  namespace mcmc {
    
    class mock_hmc: public base_hmc<mock_model,
                                    ps_point,
                                    mock_hamiltonian,
                                    mock_integrator,
                                    rng_t> {
      
    public:
      
      mock_hmc(mock_model& m, rng_t& rng, std::ostream* o, std::ostream* e) : 
        base_hmc<mock_model,ps_point,mock_hamiltonian,mock_integrator,rng_t>
        (m, rng, o, e)
      { this->name_ = "Mock HMC"; }
      
      
      sample transition(sample& init_sample) {
        this->seed(init_sample.cont_params());
        return sample(this->z_.q, - this->hamiltonian_.V(this->z_), 0);
      }
      
      void write_sampler_param_names(std::ostream& o) {};
      
      void write_sampler_params(std::ostream& o) {};
      
      void get_sampler_param_names(std::vector<std::string>& names) {};
      
      void get_sampler_params(std::vector<double>& values) {};
      
    };
    
  }
  
}

TEST(McmcBaseHMC, point_construction) {

  rng_t base_rng(0);
  
  Eigen::VectorXd q(2);
  q(0) = 5;
  q(1) = 1;

  std::stringstream output, error;

  stan::mcmc::mock_model model(q.size());  
  stan::mcmc::mock_hmc sampler(model, base_rng, &output, &error);

  EXPECT_EQ(q.size(), sampler.z().q.size());
  EXPECT_EQ(static_cast<int>(q.size()), sampler.z().g.size());

  EXPECT_EQ("", output.str());
  EXPECT_EQ("", error.str());
}

TEST(McmcBaseHMC, seed) {
  
  rng_t base_rng(0);
  
  Eigen::VectorXd q(2);
  q(0) = 5;
  q(1) = 1;
  
  std::stringstream output, error;

  stan::mcmc::mock_model model(q.size());  
  stan::mcmc::mock_hmc sampler(model, base_rng, &output, &error);

  sampler.seed(q);
  
  for (int i = 0; i < q.size(); ++i)
    EXPECT_EQ(q(i), sampler.z().q(i));
  
}

TEST(McmcBaseHMC, set_nominal_stepsize) {
  
  rng_t base_rng(0);
  
  Eigen::VectorXd q(2);
  q(0) = 5;
  q(1) = 1;
  
  std::stringstream output, error;

  stan::mcmc::mock_model model(q.size());  
  stan::mcmc::mock_hmc sampler(model, base_rng, &output, &error);
  
  double old_epsilon = 1.0;
  sampler.set_nominal_stepsize(old_epsilon);
  EXPECT_EQ(old_epsilon, sampler.get_nominal_stepsize());
  
  sampler.set_nominal_stepsize(-0.1);
  EXPECT_EQ(old_epsilon, sampler.get_nominal_stepsize());
  
  EXPECT_EQ("", output.str());
  EXPECT_EQ("", error.str());
}

TEST(McmcBaseHMC, set_stepsize_jitter) {
  
  rng_t base_rng(0);
  
  Eigen::VectorXd q(2);
  q(0) = 5;
  q(1) = 1;
  
  std::stringstream output, error;

  stan::mcmc::mock_model model(q.size());  
  stan::mcmc::mock_hmc sampler(model, base_rng, &output, &error);
  
  double old_jitter = 0.1;
  sampler.set_stepsize_jitter(old_jitter);
  EXPECT_EQ(old_jitter, sampler.get_stepsize_jitter());
  
  sampler.set_nominal_stepsize(-0.1);
  EXPECT_EQ(old_jitter, sampler.get_stepsize_jitter());
  
  EXPECT_EQ("", output.str());
  EXPECT_EQ("", error.str());
}

