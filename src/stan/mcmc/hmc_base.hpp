#ifndef __STAN__MCMC__HMCBASE__BETA__
#define __STAN__MCMC__HMCBASE__BETA__

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_01.hpp>

#include <stan/model/prob_grad.hpp>

#include <stan/mcmc/mcmc_base.hpp>
#include <stan/mcmc/util.hpp>

namespace stan {

  namespace mcmc {

    class ps_point {

    public:
    
      ps_point(int n): q(n), r(n), p(Eigen::VectorXd::Zero(n)) {};
        
      std::vector<double> q;
      std::vector<int> r;
      Eigen::VectorXd p;

    };

    template <class M, template<class> class H, template<class> class I, 
              class BaseRNG = boost::mt19937>
    class hmc_base: public mcmc_sampler {
    
    public:
    
      hmc_base(M &m, BaseRNG rng = BaseRNG(std::time(0)));
      
    protected:
    
      I<H<M> > _integrator;
      H<M> _hamiltonian;
      
      BaseRNG _rng;
      
      // Normal(0, 1) RNG
      boost::variate_generator<BaseRNG&, boost::normal_distribution<> > _rand_unit_gaus;
      
      // Uniform(0, 1) RNG
      boost::uniform_01<BaseRNG&> _rand_uniform;                
    
    };

    template <class M, template<class> class H, template<class> class I, class BaseRNG>
    hmc_base<M, H, I, BaseRNG>::hmc_base(
                                         M &m, 
                                         BaseRNG rng)
    : mcmc_sampler(),
    _hamiltonian(m), 
    _rng(rng),
    _rand_unit_gaus(rng, boost::normal_distribution<>()),
    _rand_uniform(rng)
    {};
    
  } // mcmc

} // stan

#endif
