#ifndef STAN__MCMC__BASE__HAMILTONIAN__BETA
#define STAN__MCMC__BASE__HAMILTONIAN__BETA

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
      
      base_hamiltonian(M& m, std::ostream* e): model_(m), err_stream_(e) {};
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
          stan::model::gradient(model_, z.q, z.V, z.g, err_stream_);
          z.V *= -1;
        } catch (const std::exception& e) {
          this->write_error_msg_(err_stream_, e);
          z.V = std::numeric_limits<double>::infinity();
        }
        
        z.g *= -1;
        
      }
      
    protected: 
      
        M& model_;
      
        std::ostream* err_stream_;
      
        void write_error_msg_(std::ostream* error_msgs,
                              const std::exception& e) {
          
          if (!error_msgs) return;
          
          *error_msgs << std::endl
                      << "Informational Message: The current Metropolis proposal is about to be "
                      << "rejected because of the following issue:"
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
