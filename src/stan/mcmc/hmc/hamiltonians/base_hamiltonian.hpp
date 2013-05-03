#ifndef __STAN__MCMC__BASE__HAMILTONIAN__BETA__
#define __STAN__MCMC__BASE__HAMILTONIAN__BETA__

#include <stdexcept>
#include <fstream>
#include <iostream>
#include <vector>

#include <stan/math/matrix/Eigen.hpp>

namespace stan {

  namespace mcmc {

    template <typename M, typename P, typename BaseRNG>
    class base_hamiltonian {
      
    public:
      
      base_hamiltonian(M& m, std::ostream* e): _model(m), _err_stream(e) {};
      ~base_hamiltonian() {}; 
      
      virtual double T(P& z) = 0;
      double V(P& z) { return z.V; }
      
      virtual double tau(P& z) = 0;
      virtual double phi(P& z) = 0;
      
      double H(P& z) { return T(z) + V(z); }
      
      // tau = 0.5 p_{i} p_{j} Lambda^{ij} (q) 
      virtual const Eigen::VectorXd dtau_dq(P& z) = 0;
      virtual const Eigen::VectorXd dtau_dp(P& z) = 0;
      
      // phi = 0.5 * log | Lambda (q) | + V(q)
      virtual const Eigen::VectorXd dphi_dq(P& z) = 0;
      
      virtual void sample_p(P& z, BaseRNG& rng) = 0;
      
      virtual void init(P& z) { this->update(z); }
      
      virtual void update(P& z) {
        
        std::vector<double> grad_lp(this->_model.num_params_r());
        
        try {
          z.V = - this->_model.grad_log_prob(z.q, z.r, grad_lp, _err_stream);
        } catch (std::domain_error e) {
          this->_write_error_msg(_err_stream, e);
          z.V = std::numeric_limits<double>::infinity();
        }
        
        Eigen::Map<Eigen::VectorXd> eigen_g(&(grad_lp[0]), grad_lp.size());
        z.g = - eigen_g;
        
      }
      
    protected: 
      
        M& _model;
      
        std::ostream* _err_stream;
      
        void _write_error_msg(std::ostream* error_msgs,
                             const std::domain_error& e) {
          
          if (!error_msgs) return;
          
          *error_msgs << std::endl
                      << "Informational Message: The parameter state is about to be Metropolis"
                      << " rejected due to the following underlying, non-fatal (really)"
                      << " issue (and please ignore that what comes next might say 'error'): "
                      << e.what()
                      << std::endl
                      << "If the problem persists across multiple draws, you might have"
                      << " a problem with an initial state or a gradient somewhere."
                      << std::endl
                      << " If the problem does not persist, the resulting samples will still"
                      << " be drawn from the posterior."
                      << std::endl;
          
      }
      
    };
    
  } // mcmc

} // stan
          

#endif
