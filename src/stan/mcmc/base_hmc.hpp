#ifndef __STAN__MCMC__BASE__HMC__BETA__
#define __STAN__MCMC__BASE__HMC__BETA__

#include <boost/random/variate_generator.hpp>
#include <boost/random/uniform_01.hpp>

#include <stan/mcmc/base_mcmc.hpp>

namespace stan {

  namespace mcmc {

    template <class M, class P, template<class, class> class H, 
              template<class, class> class I, class BaseRNG>
    class base_hmc: public base_mcmc {
    
    public:
    
      base_hmc(M &m, BaseRNG& rng): base_mcmc(),
                                    _hamiltonian(m), 
                                    _rand_int(rng),
                                    _rand_uniform(_rand_int)
      {};
      
    protected:
    
      I<H<M, BaseRNG>, P> _integrator;
      H<M, BaseRNG> _hamiltonian;
      
      BaseRNG& _rand_int;
      
      // Uniform(0, 1) RNG
      boost::uniform_01<BaseRNG&> _rand_uniform;                
    
    };
    
  } // mcmc

} // stan

#endif
