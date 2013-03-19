#ifndef __STAN__MCMC__HAMILTONIAN__BETA__
#define __STAN__MCMC__HAMILTONIAN__BETA__

#include <stan/model/prob_grad.hpp>

#include <stan/mcmc/hmc_base.hpp>
#include <stan/mcmc/util.hpp>

namespace stan {

  namespace mcmc {

    // Base Hamiltonian
    template <typename M>
    class base_hamiltonian {
      
    public:
      
      base_hamiltonian(M& m): _model(m) {};
      ~base_hamiltonian() {}; 
      
      virtual double T(psPoint& z) = 0;
      double V(psPoint& z) { - _model.log_prob(z.q, NULL); }
      
      virtual double tau(psPoint& z) = 0;
      virtual double phi(psPoint& z) = 0;
      
      double H(psPoint& z) { return T(z) + V(z); }
      
      // tau = 0.5 p_{i} p_{j} Lambda^{ij} (q) 
      virtual Eigen::VectorXd dtau_dq(psPoint& z) = 0;
      virtual Eigen::VectorXd dtau_dp(psPoint& z) = 0;
      
      // phi = 0.5 * log | Lambda (q) | + V(q)
      virtual Eigen::VectorXd dphi_dq(psPoint& z) = 0;
      
      virtual void sampleP(psPoint& z, VectorXd& rand_unit_gaus) = 0;
      
    private: 
      
        M _model;
      
    };
    
    // Euclidean Manifold with Unit Metric
    template <typename M>
    class unit_metric: public base_hamiltonian<M> {
      
    public:
      
      unit_metric(M& m): base_hamiltonian(m), _g(m.num_params_r()) {};
      ~unit_metric() {};
      
      double T(psPoint& z);
      
      double tau(psPoint& z) { return T(z); }
      double phi(psPoint& z) { return V(z); }
      
      Eigen::VectorXd dtau_dq(psPoint& z);
      Eigen::VectorXd dtau_dp(psPoint& z);
      
      Eigen::VectorXd dphi_dq(psPoint& z);
      
      void sampleP(psPoint& z, VectorXd& rand_unit_gaus);      
      
    private:
      
      Eigen::VectorXd _g;
      
    };
    
    template <typename M>
    double unit_metric<M>::T(psPoint& z) {
      return 0.5 * z.p.squaredNorm();
    }
    
    template <typename M>
    double unit_metric<M>::dtau_dq(psPoint& z) {
      return VectorXd::Zero(_model.num_params_r());
    }
    
    template <typename M>
    double unit_metric<M>::dtau_dp(psPoint& z) {
      return z.p();
    }
    
    template <typename M>
    double unit_metric<M>::dphi_dq(psPoint& z) {
      _model.grad_log_prog(z.q, NULL, g);
      return g;
    }
    
    template <typename M>
    double unit_metric<M>::sampleP(psPoint& z, VectorXd& rand_unit_gaus) {
      z.p = rand_unit_gaus;
    }
    
    // DEFINE HAMILTONIANS HERE
    
    // Base Hamiltonian
    
    // Unit Mass
    
    // Diag Mass
    
    // Dense Mass
    
    // SoftAbs

  } // mcmc

} // stan
          

#endif
