#include <test/unit/mcmc/hmc/mock_hmc.hpp>
#include <stan/mcmc/hmc/nuts/base_nuts.hpp>

#include <boost/random/additive_combine.hpp>

#include <gtest/gtest.h>

typedef boost::ecuyer1988 rng_t;

namespace stan {
  
  namespace mcmc {
    
    class mock_nuts: public base_nuts<mock_model,
                                      ps_point,
                                      mock_hamiltonian,
                                      mock_integrator,
                                      rng_t> {
      
    public:
      
      mock_nuts(mock_model &m, rng_t& rng, std::ostream* o, std::ostream* e)
        : base_nuts<mock_model,ps_point,mock_hamiltonian,mock_integrator,rng_t>(m, rng, o, e)
      { this->_name = "Mock NUTS"; }
      
    private:
      
      bool compute_criterion(ps_point& start,
                             ps_point& finish,
                             Eigen::VectorXd& rho) { return true; }
      
    };
    
  }
  
}

TEST(McmcBaseNuts, set_max_depth) {
  
  rng_t base_rng(0);

  Eigen::VectorXd q(2);
  q(0) = 5;
  q(1) = 1;
  
  stan::mcmc::mock_model model(q.size());
  
  stan::mcmc::mock_nuts sampler(model, base_rng, &std::cout, &std::cerr);
  
  int old_max_depth = 1;
  sampler.set_max_depth(old_max_depth);
  EXPECT_EQ(old_max_depth, sampler.get_max_depth());
  
  sampler.set_max_depth(-1);
  EXPECT_EQ(old_max_depth, sampler.get_max_depth());
  
}


TEST(McmcBaseNuts, set_max_delta) {
  
  rng_t base_rng(0);
  
  Eigen::VectorXd q(2);
  q(0) = 5;
  q(1) = 1;
  
  stan::mcmc::mock_model model(q.size());
  
  stan::mcmc::mock_nuts sampler(model, base_rng, &std::cout, &std::cerr);
  
  double old_max_delta = 10;
  sampler.set_max_delta(old_max_delta);
  EXPECT_EQ(old_max_delta, sampler.get_max_delta());
  
}

TEST(McmcBaseNuts, build_tree) {
  
  rng_t base_rng(0);
  
  int model_size = 1;
  double init_momentum = 1.5;
  
  Eigen::VectorXd rho = Eigen::VectorXd::Zero(model_size);
  
  stan::mcmc::ps_point z_init(model_size);
  z_init.q(0) = 0;
  z_init.p(0) = init_momentum;
  
  stan::mcmc::ps_point z_propose(model_size);
  
  stan::mcmc::nuts_util util;
  util.log_u = -1;
  util.H0 = -0.1;
  util.sign = 1;
  util.n_tree = 0;
  util.sum_prob = 0;
  
  stan::mcmc::mock_model model(model_size);
  stan::mcmc::mock_nuts sampler(model, base_rng, &std::cout, &std::cerr);
  
  sampler.set_nominal_stepsize(1);
  sampler.set_stepsize_jitter(0);
  sampler.sample_stepsize();
  sampler.z() = z_init;
  
  int n_valid = sampler.build_tree(3, rho, &z_init, z_propose, util);
  
  EXPECT_EQ(8, n_valid);
  
  EXPECT_EQ(8, util.n_tree);
  EXPECT_FLOAT_EQ(std::exp(util.H0) * util.n_tree, util.sum_prob);
  
  EXPECT_EQ(init_momentum * util.n_tree, rho(0));
  
  EXPECT_EQ(init_momentum, z_init.q(0));
  EXPECT_EQ(init_momentum, z_init.p(0));
  
  EXPECT_EQ(8 * init_momentum, sampler.z().q(0));
  EXPECT_EQ(init_momentum, sampler.z().p(0));
  
}
