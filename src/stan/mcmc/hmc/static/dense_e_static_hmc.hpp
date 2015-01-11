#ifndef STAN__MCMC__DENSE__E__STATIC__HMC__BETA
#define STAN__MCMC__DENSE__E__STATIC__HMC__BETA

#include <stan/mcmc/hmc/static/base_static_hmc.hpp>
#include <stan/mcmc/hmc/hamiltonians/dense_e_point.hpp>
#include <stan/mcmc/hmc/hamiltonians/dense_e_metric.hpp>
#include <stan/mcmc/hmc/integrators/expl_leapfrog.hpp>

namespace stan {

  namespace mcmc {

    // Hamiltonian Monte Carlo on a 
    // Euclidean manifold with dense metric
    // and static integration time
    
    template <typename M, class BaseRNG>
    class dense_e_static_hmc: public base_static_hmc<M, 
                                                     dense_e_point,
                                                     dense_e_metric, 
                                                     expl_leapfrog, 
                                                     BaseRNG> {
      
    public:
      
      dense_e_static_hmc(M &m, BaseRNG& rng, std::ostream* o = &std::cout, std::ostream* e = 0):
      base_static_hmc<M, dense_e_point, dense_e_metric, expl_leapfrog, BaseRNG>(m, rng, o, e)
      { this->name_ = "Static HMC with a dense Euclidean metric"; }
                                            
                        
    };

  } // mcmc

} // stan
          

#endif
