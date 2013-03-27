#ifndef __STAN__MCMC__HAMILTONIAN__BETA__
#define __STAN__MCMC__HAMILTONIAN__BETA__

#include <stan/mcmc/hmc_base.hpp>
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
      
      virtual double T(ps_point& z) = 0;
      double V(ps_point& z) { return - _model.log_prob(z.q, z.r); }
      
      virtual double tau(ps_point& z) = 0;
      virtual double phi(ps_point& z) = 0;
      
      double H(ps_point& z) { return T(z) + V(z); }
      
      // tau = 0.5 p_{i} p_{j} Lambda^{ij} (q) 
      virtual const Eigen::VectorXd dtau_dq(ps_point& z) = 0;
      virtual const Eigen::VectorXd dtau_dp(ps_point& z) = 0;
      
      // phi = 0.5 * log | Lambda (q) | + V(q)
      virtual const Eigen::VectorXd dphi_dq(ps_point& z) = 0;
      
      virtual void sampleP(ps_point& z, Eigen::VectorXd& rand_unit_gaus) = 0;
      
      virtual void init(ps_point& z) { this->update(z); }
      
      virtual void update(ps_point& z) {}; // Default no-op
      
    protected: 
      
        M& _model;
      
    };
    
    // Euclidean Manifold with Unit Metric
    template <typename M>
    class unit_metric: public base_hamiltonian<M> {
      
    public:
      
      unit_metric(M& m): base_hamiltonian<M>(m), _g(m.num_params_r()) {};
      ~unit_metric() {};
      
      double T(ps_point& z);
      
      double tau(ps_point& z) { return T(z); }
      double phi(ps_point& z) { return this->V(z); }
      
      const Eigen::VectorXd dtau_dq(ps_point& z);
      const Eigen::VectorXd dtau_dp(ps_point& z);
      
      const Eigen::VectorXd dphi_dq(ps_point& z);
      
      void sampleP(ps_point& z, Eigen::VectorXd& rand_unit_gaus);   
      
      void update(ps_point& z);
      
    private:
      
      double _V;
      Eigen::VectorXd _g;
      
    };
    
    template <typename M>
    double unit_metric<M>::T(ps_point& z) {
      return 0.5 * z.p.squaredNorm();
    }
    
    template <typename M>
    const Eigen::VectorXd unit_metric<M>::dtau_dq(ps_point& z) {
      return Eigen::VectorXd::Zero(this->_model.num_params_r());
    }
    
    template <typename M>
    const Eigen::VectorXd unit_metric<M>::dtau_dp(ps_point& z) {
      return z.p;
    }
    
    template <typename M>
    const Eigen::VectorXd unit_metric<M>::dphi_dq(ps_point& z) {
      return _g;
    }
    
    template <typename M>
    void unit_metric<M>::sampleP(ps_point& z, Eigen::VectorXd& rand_unit_gaus) {
      z.p = rand_unit_gaus;
    }
    
    template <typename M>
    void unit_metric<M>::update(ps_point& z) {
      std::vector<double> grad_lp(this->_model.num_params_r());
      _V = this->_model.grad_log_prob(z.q, z.r, grad_lp);
      Eigen::Map<Eigen::VectorXd> eigen_g(&(grad_lp[0]), grad_lp.size());
      _g = - eigen_g;
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
