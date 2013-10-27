#ifndef __STAN__MCMC__SOFTABS__NUTS__BETA__
#define __STAN__MCMC__SOFTABS__NUTS__BETA__

#include <stan/mcmc/hmc/nuts/base_nuts.hpp>
#include <stan/mcmc/hmc/hamiltonians/softabs_point.hpp>
#include <stan/mcmc/hmc/hamiltonians/softabs_metric.hpp>
#include <stan/mcmc/hmc/integrators/impl_leapfrog.hpp>

namespace stan {

  namespace mcmc {

    // The No-U-Turn Sampler (NUTS) on a
    // Riemannain manifold wtih the SoftAbs metric
    
    template <typename M, class BaseRNG>
    class softabs_nuts: public base_nuts<M,
                                         softabs_point,
                                         softabs_metric,
                                         impl_leapfrog,
                                         BaseRNG> {
      
    public:
      
    softabs_nuts(M &m, BaseRNG& rng, std::ostream* o = &std::cout, std::ostream* e = 0):
    base_nuts<M, softabs_point, softabs_metric, impl_leapfrog, BaseRNG>(m, rng, o, e)
    { this->_name = "NUTS with the SoftAbs Riemannian metric"; }
      
    void test(sample& s) {
      
      this->seed(s.cont_params(), s.disc_params());
      this->_hamiltonian.sample_p(this->_z, this->_rand_int);
      
      this->_hamiltonian.test_derivatives(this->_z);
    }
                                           
    private:
      
      bool _compute_criterion(ps_point& start, 
                              softabs_point& finish,
                              Eigen::VectorXd& rho) {
        
        bool end_check = (rho.dot( this->_hamiltonian.metric_inv_dot_p(finish) )
                          - finish.p.dot( this->_hamiltonian.metric_inv_dot_p(finish) ) ) > 0;
        
        std::vector<double> q_swap = finish.q;
        Eigen::VectorXd p_swap = finish.p;
        
        finish.q = start.q;
        finish.p = start.p;
        
        this->_hamiltonian.compute_metric(finish);
        
        bool start_check = (rho.dot( this->_hamiltonian.metric_inv_dot_p(finish) )
                            - finish.p.dot( this->_hamiltonian.metric_inv_dot_p(finish) ) ) > 0;
        
        finish.q = q_swap;
        finish.p = p_swap;
        
        this->_hamiltonian.compute_metric(finish);
        
        return end_check && start_check;

      }
                                          
    };
    
  } // mcmc
    
} // stan

#endif
