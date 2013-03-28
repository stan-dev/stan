#ifndef __STAN__MCMC__BASE__HAMILTONIAN__BETA__
#define __STAN__MCMC__BASE__HAMILTONIAN__BETA__

#include <Eigen/Dense>

namespace stan {

  namespace mcmc {

    template <typename M, typename P>
    class base_hamiltonian {
      
    public:
      
      base_hamiltonian(M& m): _model(m) {};
      ~base_hamiltonian() {}; 
      
      virtual double T(P& z) = 0;
      double V(P& z) { return - _model.log_prob(z.q, z.r); }
      
      virtual double tau(P& z) = 0;
      virtual double phi(P& z) = 0;
      
      double H(P& z) { return T(z) + V(z); }
      
      // tau = 0.5 p_{i} p_{j} Lambda^{ij} (q) 
      virtual const Eigen::VectorXd dtau_dq(P& z) = 0;
      virtual const Eigen::VectorXd dtau_dp(P& z) = 0;
      
      // phi = 0.5 * log | Lambda (q) | + V(q)
      virtual const Eigen::VectorXd dphi_dq(P& z) = 0;
      
      virtual void sampleP(P& z, Eigen::VectorXd& rand_unit_gaus) = 0;
      
      virtual void init(P& z) { this->update(z); }
      
      virtual void update(P& z) {}; // Default no-op
      
    protected: 
      
        M& _model;
      
    };
    
  } // mcmc

} // stan
          

#endif
