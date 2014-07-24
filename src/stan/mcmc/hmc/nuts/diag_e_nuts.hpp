#ifndef STAN__MCMC__DIAG__E__NUTS__BETA
#define STAN__MCMC__DIAG__E__NUTS__BETA

#include <stan/mcmc/hmc/nuts/base_nuts.hpp>
#include <stan/mcmc/hmc/hamiltonians/diag_e_point.hpp>
#include <stan/mcmc/hmc/hamiltonians/diag_e_metric.hpp>
#include <stan/mcmc/hmc/integrators/expl_leapfrog.hpp>

namespace stan {

  namespace mcmc {

    // The No-U-Turn Sampler (NUTS) on a
    // Euclidean manifold with diagonal metric
    
    template <typename M, class BaseRNG>
    class diag_e_nuts: public base_nuts<M,
                                        diag_e_point,
                                        diag_e_metric,
                                        expl_leapfrog,
                                        BaseRNG> {
      
    public:
      
      diag_e_nuts(M &m, BaseRNG& rng, std::ostream* o = &std::cout, std::ostream* e = 0):
      base_nuts<M, diag_e_point, diag_e_metric, expl_leapfrog, BaseRNG>(m, rng, o, e)
      { this->name_ = "NUTS with a diagonal Euclidean metric"; }
      
      // Note that the points don't need to be swapped
      // here since start.mInv = finish.mInv
      bool compute_criterion(ps_point& start, 
                             diag_e_point& finish,
                             Eigen::VectorXd& rho) {
        return finish.mInv.cwiseProduct(finish.p).dot(rho - finish.p) > 0
               && finish.mInv.cwiseProduct(start.p).dot(rho - start.p) > 0;
      }
                                          
    };
    
  } // mcmc
    
} // stan

#endif
