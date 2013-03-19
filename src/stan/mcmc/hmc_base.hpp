#ifndef __STAN__MCMC__HAMILTONIAN__BETA__
#define __STAN__MCMC__HAMILTONIAN__BETA__

#include <ctime>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_01.hpp>

#include <stan/mcmc/adaptive_sampler.hpp>
#include <stan/model/prob_grad.hpp>
#include <stan/mcmc/util.hpp>

namespace stan {

  namespace mcmc {

    class psPoint {

    public:
    
      psPoint(int n): q(Eigen::VectorXd::Zero(n)), p(Eigen::VectorXd::Zero(n)) {};
        
      Eigen::VectorXd q;
      Eigen::VectorXd p;

    };

    template <typename M, typename H, typename I, 
              class BaseRNG = boost::mt19937>
    class hmc_base
    {
    
    public:
    
      hmc_base(M &m, BaseRNG rng = BaseRNG(std::time(0))):hamiltonian(m), _rng(rng) {};
      
      virtual void sample(psPoint& z, Eigen::VectorXd& rand_unit_gaus) = 0;
    
    private:
    
      I<H> _integrator;
      H _hamiltonian;
      
      BaseRNG _rng;
      
      // Normal(0, 1) RNG
      boost::variate_generator<BaseRNG&, boost::normal_distribution<> > _rand_unit_gaus;
      
      // Uniform(0, 1) RNG
      boost::uniform_01<BaseRNG&> _rand_uniform;                
    
    }

    <template typename M, typename H, typename I<H>, class BaseRNG = boost::mt19937>
    hmc::base<M, H, I, BaseRNG>::hmc_base(
                                             M &m, 
                                             BaseRNG rng = BaseRNG(std::time(0)))
    :hamiltonian(m), 
    _rng(rng),
    _rand_unit_gaus(rng, boost::normal_distribution<>()),
    _rand_uniform(rng)
    {};
    
  } // mcmc

} // stan
          

#endif
