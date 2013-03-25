#ifndef __STAN__MCMC__HAMILTONIAN__BETA__
#define __STAN__MCMC__HAMILTONIAN__BETA__

#include <stan/mcmc/util.hpp>

#include <Eigen/Dense>

namespace stan {

  namespace mcmc {

    // Base Hamiltonian
    template <typename M>
    class base_hamiltonian {
      
    public:
      
      base_hamiltonian(M& m): _model(m) {};
      ~base_hamiltonian() {}; 
      
      virtual double T(psPoint& z) = 0;
      double V(psPoint& z) { return - _model.log_prob(z.q, z.r); }
      
      virtual double tau(psPoint& z) = 0;
      virtual double phi(psPoint& z) = 0;
      
      double H(psPoint& z) { return T(z) + V(z); }
      
      // tau = 0.5 p_{i} p_{j} Lambda^{ij} (q) 
      virtual Eigen::VectorXd dtau_dq(psPoint& z) = 0;
      virtual Eigen::VectorXd dtau_dp(psPoint& z) = 0;
      
      // phi = 0.5 * log | Lambda (q) | + V(q)
      virtual Eigen::VectorXd dphi_dq(psPoint& z) = 0;
      
      virtual void sampleP(psPoint& z, Eigen::VectorXd& rand_unit_gaus) = 0;
      
    protected: 
      
        M& _model;
      
    };
    
    // Euclidean Manifold with Unit Metric
    template <typename M>
    class unit_metric: public base_hamiltonian<M> {
      
    public:
      
      unit_metric(M& m): base_hamiltonian<M>(m), _g(m.num_params_r()) {};
      ~unit_metric() {};
      
      double T(psPoint& z);
      
      double tau(psPoint& z) { return T(z); }
      double phi(psPoint& z) { return this->V(z); }
      
      Eigen::VectorXd dtau_dq(psPoint& z);
      Eigen::VectorXd dtau_dp(psPoint& z);
      
      Eigen::VectorXd dphi_dq(psPoint& z);
      
      void sampleP(psPoint& z, Eigen::VectorXd& rand_unit_gaus);      
      
    private:
      
      Eigen::VectorXd _g;
      
    };
    
    template <typename M>
    double unit_metric<M>::T(psPoint& z) {
      return 0.5 * z.p.squaredNorm();
    }
    
    template <typename M>
    Eigen::VectorXd unit_metric<M>::dtau_dq(psPoint& z) {
      return Eigen::VectorXd::Zero(this->_model.num_params_r());
    }
    
    template <typename M>
    Eigen::VectorXd unit_metric<M>::dtau_dp(psPoint& z) {
      return z.p;
    }
    
    template <typename M>
    Eigen::VectorXd unit_metric<M>::dphi_dq(psPoint& z) {
      
      std::vector<double> g(this->_model.num_params_r());
      this-> _model.grad_log_prob(z.q, z.r, g);
      Eigen::Map<Eigen::VectorXd> eigen_g(&g[0], g.size());
      _g = - eigen_g;
      return _g;
    }
    
    template <typename M>
    void unit_metric<M>::sampleP(psPoint& z, Eigen::VectorXd& rand_unit_gaus) {
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
