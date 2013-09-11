#ifndef __STAN__MCMC__UNIT__E__TRAJECTORY__BETA__
#define __STAN__MCMC__UNIT__E__TRAJECTORY__HMC__BETA__

#include <stan/mcmc/hmc/base_trajectory.hpp>
#include <stan/mcmc/hmc/hamiltonians/unit_e_point.hpp>
#include <stan/mcmc/hmc/hamiltonians/unit_e_metric.hpp>
#include <stan/mcmc/hmc/integrators/expl_leapfrog.hpp>

namespace stan {

  namespace mcmc {

    // Hamiltonian Monte Carlo on a 
    // Euclidean manifold with unit metric
    // and static integration time
    
    template <typename M, class BaseRNG>
    class unit_e_trajectory: public base_trajectory<M,
                                                    unit_e_point,
                                                    unit_e_metric, 
                                                    expl_leapfrog, 
                                                    BaseRNG> {
      
    public:
      
      unit_e_trajectory(M &m, BaseRNG& rng, std::ostream* o = &std::cout, std::ostream* e = 0):
      base_trajectory<M, unit_e_point, unit_e_metric, expl_leapfrog, BaseRNG>(m, rng, o, e)
      { this->_name = "Trajectory diagnostic a unit Euclidean metric"; }
                        
    };

  } // mcmc

} // stan
          

#endif
