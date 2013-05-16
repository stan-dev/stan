#ifndef __STAN__MCMC__UNIT__E__NUTS__BETA__
#define __STAN__MCMC__UNIT__E__NUTS__BETA__

#include <stan/mcmc/hmc/nuts/base_nuts.hpp>
#include <stan/mcmc/hmc/hamiltonians/unit_e_point.hpp>
#include <stan/mcmc/hmc/hamiltonians/unit_e_metric.hpp>
#include <stan/mcmc/hmc/integrators/expl_leapfrog.hpp>

namespace stan {

  namespace mcmc {

    // The No-U-Turn Sampler (NUTS) on a
    // Euclidean manifold with unit metric
    
    template <typename M, class BaseRNG>
    class unit_e_nuts: public base_nuts<M,
                                        unit_e_point,
                                        unit_e_metric,
                                        expl_leapfrog,
                                        BaseRNG> {
      
    public:
      
    unit_e_nuts(M &m, BaseRNG& rng, std::ostream* o = &std::cout, std::ostream* e = 0):
    base_nuts<M, unit_e_point, unit_e_metric, expl_leapfrog, BaseRNG>(m, rng, o, e)
    { this->_name = "NUTS with a unit Euclidean metric"; }
      
    private:
      
      bool _compute_criterion(ps_point& start, 
                              unit_e_point& finish, 
                              Eigen::VectorXd& rho) {
        return rho.dot(finish.p) > 0
            && rho.dot(start.p)  > 0;
      }
                                          
    };
    
  } // mcmc
    
} // stan

#endif
