#ifndef __STAN__MCMC__HAMILTONIAN__BETA__
#define __STAN__MCMC__HAMILTONIAN__BETA__

#include <stan/mcmc/hmc_base.hpp>
#include <stan/mcmc/util.hpp>

#include <Eigen/Dense>

namespace stan {

  namespace mcmc {

    // Base Hamiltonian
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
    
    // Euclidean Manifold with Unit Metric
    class unit_e_point: public ps_point {
      
    public:
      
      unit_e_point(int n, int m): ps_point(n, m), V(0), g(Eigen::VectorXd::Zero(n)) {};
      
      double V;
      Eigen::VectorXd g;
      
    };
    
    template <typename M>
    class unit_metric: public base_hamiltonian<M, unit_e_point> {
      
    public:
      
      unit_metric(M& m): base_hamiltonian<M, unit_e_point>(m) {};
      ~unit_metric() {};
      
      double T(unit_e_point& z);
      
      double tau(unit_e_point& z) { return T(z); }
      double phi(unit_e_point& z) { return this->V(z); }
      
      const Eigen::VectorXd dtau_dq(unit_e_point& z);
      const Eigen::VectorXd dtau_dp(unit_e_point& z);
      
      const Eigen::VectorXd dphi_dq(unit_e_point& z);
      
      void sampleP(unit_e_point& z, Eigen::VectorXd& rand_unit_gaus);   
      
      void update(unit_e_point& z);
      
    private:
      
    };
    
    template <typename M>
    double unit_metric<M>::T(unit_e_point& z) {
      return 0.5 * z.p.squaredNorm();
    }
    
    template <typename M>
    const Eigen::VectorXd unit_metric<M>::dtau_dq(unit_e_point& z) {
      return Eigen::VectorXd::Zero(this->_model.num_params_r());
    }
    
    template <typename M>
    const Eigen::VectorXd unit_metric<M>::dtau_dp(unit_e_point& z) {
      return z.p;
    }
    
    template <typename M>
    const Eigen::VectorXd unit_metric<M>::dphi_dq(unit_e_point& z) {
      return z.g;
    }
    
    template <typename M>
    void unit_metric<M>::sampleP(unit_e_point& z, Eigen::VectorXd& rand_unit_gaus) {
      z.p = rand_unit_gaus;
    }
    
    template <typename M>
    void unit_metric<M>::update(unit_e_point& z) {
      std::vector<double> grad_lp(this->_model.num_params_r());
      z.V = - this->_model.grad_log_prob(z.q, z.r, grad_lp);
      Eigen::Map<Eigen::VectorXd> eigen_g(&(grad_lp[0]), grad_lp.size());
      z.g = - eigen_g;
    }
    
    // DEFINE HAMILTONIANS HERE
    
    // Diag Mass
    
    // Dense Mass
    
    // SoftAbs

  } // mcmc

} // stan
          

#endif
