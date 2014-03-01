#ifndef __STAN__MCMC__BASE__HAMILTONIAN__BETA__
#define __STAN__MCMC__BASE__HAMILTONIAN__BETA__

#include <stdexcept>
#include <fstream>
#include <iostream>
#include <vector>

#include <stan/math/matrix/Eigen.hpp>
#include <stan/model/util.hpp>

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
        
        try {
          stan::model::gradient(_model, z.q, z.V, z.g, _err_stream);
          z.V *= -1;
        } catch (const std::exception& e) {
          this->_write_error_msg(_err_stream, e);
          z.V = std::numeric_limits<double>::infinity();
        }
        
        z.g *= -1;
        
      }
      
    protected: 
      
        M& _model;
      
        std::ostream* _err_stream;
      
        void _write_error_msg(std::ostream* error_msgs,
                             const std::exception& e) {
          
          if (!error_msgs) return;
          
          *error_msgs << std::endl
                      << "Informational Message: The current Metropolis proposal is about to be "
                      << "rejected becuase of the following issue:"
                      << std::endl
                      << e.what() << std::endl
                      << "If this warning occurs sporadically, such as for highly constrained "
                      << "variable types like covariance matrices, then the sampler is fine,"
                      << std::endl
                      << "but if this warning occurs often then your model may be either severely "
                      << "ill-conditioned or misspecified."
                      << std::endl;
          
      }
      
    };
    
  } // mcmc

} // stan


#endif
