#ifndef __STAN__MCMC__DIAG__E__METRIC__BETA__
#define __STAN__MCMC__DIAG__E__METRIC__BETA__

#include <boost/random/normal_distribution.hpp>

#include <stan/mcmc/base_hamiltonian.hpp>
#include <stan/mcmc/diag_e_point.hpp>

namespace stan {
  
  namespace mcmc {
    
    // Euclidean manifold with diagonal metric
    template <typename M, typename BaseRNG>
    class diag_e_metric: public base_hamiltonian<M, diag_e_point, BaseRNG> {
      
    public:
      
      diag_e_metric(M& m): base_hamiltonian<M, diag_e_point, BaseRNG>(m) {};
      ~diag_e_metric() {};
      
      double T(diag_e_point& z) {
        return 0.5 * z.p.dot( z.mInv.cwiseProduct(z.p) );
      }
      
      double tau(diag_e_point& z) { return T(z); }
      double phi(diag_e_point& z) { return this->V(z); }
      
      const Eigen::VectorXd dtau_dq(diag_e_point& z) {
        return Eigen::VectorXd::Zero(this->_model.num_params_r());
      }

      const Eigen::VectorXd dtau_dp(diag_e_point& z) {
        return z.mInv.cwiseProduct(z.p);
      }
      
      const Eigen::VectorXd dphi_dq(diag_e_point& z) {
        return z.g;
      }
      
      void sample_p(diag_e_point& z, BaseRNG& rng) {
        
        boost::variate_generator<BaseRNG&, boost::normal_distribution<> > 
          _rand_diag_gaus(rng, boost::normal_distribution<>());
        
        for (size_t i = 0; i < z.p.size(); ++i) z.p(i) = _rand_diag_gaus() / sqrt(z.mInv(i));

      }
      
      void update(diag_e_point& z) {
        std::vector<double> grad_lp(this->_model.num_params_r());
        z.V = - this->_model.grad_log_prob(z.q, z.r, grad_lp);
        Eigen::Map<Eigen::VectorXd> eigen_g(&(grad_lp[0]), grad_lp.size());
        z.g = - eigen_g;
      }
      
    };
    
  } // mcmc
  
} // stan


#endif
