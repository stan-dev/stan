#ifndef __STAN__MCMC__SOFTABS__STATIC__HMC__BETA__
#define __STAN__MCMC__SOFTABS__STATIC__HMC__BETA__

#include <stan/mcmc/hmc/static/base_static_hmc.hpp>
#include <stan/mcmc/hmc/hamiltonians/softabs_point.hpp>
#include <stan/mcmc/hmc/hamiltonians/softabs_metric.hpp>
#include <stan/mcmc/hmc/integrators/impl_leapfrog.hpp>

namespace stan {

  namespace mcmc {

    // Hamiltonian Monte Carlo on a 
    // Riemannian metric with the SoftAbs metric
    // and static integration time
    
    template <typename M, class BaseRNG>
    class softabs_static_hmc: public base_static_hmc<M,
                                                     softabs_point,
                                                     softabs_metric,
                                                     impl_leapfrog,
                                                     BaseRNG> {
      
    public:
      
      softabs_static_hmc(M &m, BaseRNG& rng, std::ostream* o = &std::cout, std::ostream* e = 0):
      base_static_hmc<M, softabs_point, softabs_metric, impl_leapfrog, BaseRNG>(m, rng, o, e)
      { this->_name = "Static HMC with the SoftAbs Riemannian metric"; }
                        
    };

  } // mcmc

} // stan
          

#endif
