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
    
      ps_point(int n, int m): q(n), r(m), p(Eigen::VectorXd::Zero(n)) {};
        
      std::vector<double> q;
      std::vector<int> r;
      Eigen::VectorXd p;

    };

    template <class M, class P, template<class> class H, 
              template<class, class> class I, class BaseRNG>
    class hmc_base: public mcmc_sampler {
    
    public:
    
      hmc_base(M &m, BaseRNG& rng);
      
    protected:
    
      I<H<M>, P> _integrator;
      H<M> _hamiltonian;
      
      BaseRNG& _rand_int;
      
      // Normal(0, 1) RNG
      boost::variate_generator<BaseRNG&, boost::normal_distribution<> > _rand_unit_gaus;
      
      // Uniform(0, 1) RNG
      boost::uniform_01<BaseRNG&> _rand_uniform;                
    
    };

    template <class M, class P, template<class> class H, 
              template<class, class> class I, class BaseRNG>
    hmc_base<M, P, H, I, BaseRNG>::hmc_base(
                                         M &m, 
                                         BaseRNG& rng)
    : mcmc_sampler(),
    _hamiltonian(m), 
    _rand_int(rng),
    _rand_unit_gaus(_rand_int, boost::normal_distribution<>()),
    _rand_uniform(_rand_int)
    {};
    
  } // mcmc

} // stan

#endif
