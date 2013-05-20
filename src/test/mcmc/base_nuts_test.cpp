#include <test/mcmc/mock_hmc.hpp>
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
      
      bool _compute_criterion(ps_point& start,
                              ps_point& finish,
                              Eigen::VectorXd& rho) { return false; }
      
    };
    
  }
  
}

TEST(McmcBaseNuts, set_max_depth) {
  
  rng_t base_rng(0);
  
  std::vector<double> q(5, 1.0);
  std::vector<int> r(2, 2);
  
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
  
  std::vector<double> q(5, 1.0);
  std::vector<int> r(2, 2);
  
  stan::mcmc::mock_model model(q.size());
  
  stan::mcmc::mock_nuts sampler(model, base_rng, &std::cout, &std::cerr);
  
  double old_max_delta = 10;
  sampler.set_max_delta(old_max_delta);
  EXPECT_EQ(old_max_delta, sampler.get_max_delta());
  
}
