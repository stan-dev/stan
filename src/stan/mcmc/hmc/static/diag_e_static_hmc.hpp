#ifndef STAN__MCMC__DIAG__E__STATIC__HMC__BETA
#define STAN__MCMC__DIAG__E__STATIC__HMC__BETA

#include <stan/mcmc/hmc/static/base_static_hmc.hpp>
#include <stan/mcmc/hmc/hamiltonians/diag_e_point.hpp>
#include <stan/mcmc/hmc/hamiltonians/diag_e_metric.hpp>
#include <stan/mcmc/hmc/integrators/expl_leapfrog.hpp>

namespace stan {

  namespace mcmc {

    // Hamiltonian Monte Carlo on a 
    // Euclidean manifold with diagonal metric
    // and static integration time
    
    template <typename M, class BaseRNG>
    class diag_e_static_hmc: public base_static_hmc<M, 
                                                    diag_e_point,
                                                    diag_e_metric, 
                                                    expl_leapfrog, 
                                                    BaseRNG> {
      
    public:
      
      diag_e_static_hmc(M &m, BaseRNG& rng, std::ostream* o = &std::cout, std::ostream* e = 0):
      base_static_hmc<M, diag_e_point, diag_e_metric, expl_leapfrog, BaseRNG>(m, rng, o, e)
      { this->name_ = "Static HMC with a diagonal Euclidean metric"; }

                        
    };

  } // mcmc

} // stan
          

#endif
