#ifndef __STAN__MCMC__UNIT__E__METRIC__BETA__
#define __STAN__MCMC__UNIT__E__METRIC__BETA__

#include <boost/random/variate_generator.hpp>
#include <boost/random/normal_distribution.hpp>

#include <stan/mcmc/hmc/hamiltonians/base_hamiltonian.hpp>
#include <stan/mcmc/hmc/hamiltonians/unit_e_point.hpp>

namespace stan {
  
  namespace mcmc {
    
    // Euclidean manifold with unit metric
    template <typename M, typename BaseRNG>
    class unit_e_metric: public base_hamiltonian<M, unit_e_point, BaseRNG> {
      
    public:
      
      unit_e_metric(M& m, std::ostream* e):
      base_hamiltonian<M, unit_e_point, BaseRNG>(m, e) {};
      ~unit_e_metric() {};
      
      double T(unit_e_point& z) {
        return 0.5 * z.p.squaredNorm();
      }
      
      double tau(unit_e_point& z) { return T(z); }
      double phi(unit_e_point& z) { return this->V(z); }
      
      const Eigen::VectorXd dtau_dq(unit_e_point& z) {
        return Eigen::VectorXd::Zero(this->_model.num_params_r());
      }

      const Eigen::VectorXd dtau_dp(unit_e_point& z) {
        return z.p;
      }
      
      const Eigen::VectorXd dphi_dq(unit_e_point& z) {
        return z.g;
      }
      
      void sample_p(unit_e_point& z, BaseRNG& rng) {
        
        boost::variate_generator<BaseRNG&, boost::normal_distribution<> > 
          _rand_unit_gaus(rng, boost::normal_distribution<>());
        
        for (size_t i = 0; i < z.p.size(); ++i) z.p(i) = _rand_unit_gaus();

      }
      
    };
    
  } // mcmc
  
} // stan


#endif
