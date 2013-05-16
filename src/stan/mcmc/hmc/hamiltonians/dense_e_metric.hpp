#ifndef __STAN__MCMC__DENSE__E__METRIC__BETA__
#define __STAN__MCMC__DENSE__E__METRIC__BETA__

#include <boost/random/variate_generator.hpp>
#include <boost/random/normal_distribution.hpp>

#include <stan/math/matrix/Eigen.hpp>
#include <Eigen/Cholesky>

#include <stan/mcmc/hmc/hamiltonians/base_hamiltonian.hpp>
#include <stan/mcmc/hmc/hamiltonians/dense_e_point.hpp>

namespace stan {
  
  namespace mcmc {
    
    // Euclidean manifold with dense metric
    template <typename M, typename BaseRNG>
    class dense_e_metric: public base_hamiltonian<M, dense_e_point, BaseRNG> {
      
    public:
      
      dense_e_metric(M& m, std::ostream* e):
      base_hamiltonian<M, dense_e_point, BaseRNG>(m, e) {};
      ~dense_e_metric() {};
      
      double T(dense_e_point& z) {
        return 0.5 * z.p.transpose() * z.mInv * z.p;
      }
      
      double tau(dense_e_point& z) { return T(z); }
      double phi(dense_e_point& z) { return this->V(z); }
      
      const Eigen::VectorXd dtau_dq(dense_e_point& z) {
        return Eigen::VectorXd::Zero(this->_model.num_params_r());
      }

      const Eigen::VectorXd dtau_dp(dense_e_point& z) {
        return z.mInv * z.p;
      }
      
      const Eigen::VectorXd dphi_dq(dense_e_point& z) {
        return z.g;
      }
      
      void sample_p(dense_e_point& z, BaseRNG& rng) {
        
        boost::variate_generator<BaseRNG&, boost::normal_distribution<> > 
          _rand_dense_gaus(rng, boost::normal_distribution<>());
        
        Eigen::VectorXd u(z.p.size());
        
        for (size_t i = 0; i < u.size(); ++i) u(i) = _rand_dense_gaus();

        z.p = z.mInv.llt().matrixL().solve(u);
        
      }
      
    };
    
  } // mcmc
  
} // stan


#endif
