#ifndef __STAN__MCMC__MOCK__HMC_BETA__
#define __STAN__MCMC__MOCK__HMC__BETA__

#include <stan/math/matrix/Eigen.hpp>

#include <stan/model/prob_grad.hpp>

#include <stan/mcmc/hmc/hamiltonians/ps_point.hpp>
#include <stan/mcmc/hmc/hamiltonians/base_hamiltonian.hpp>
#include <stan/mcmc/hmc/integrators/base_integrator.hpp>

//#include <stan/mcmc/hmc/nuts/base_nuts.hpp>

namespace stan {
  
  namespace mcmc {
    

    // Mock Model
    class mock_model: public model::prob_grad {
    public:
      
      mock_model(size_t num_params_r): model::prob_grad(num_params_r) {};
      
      template <bool propto, bool jacobian_adjust_transforms, typename T>
      double log_prob(std::vector<T>& params_r,
                      std::vector<int>& params_i,
                      std::ostream* output_stream = 0) const { 
        return 0; 
      }

      // template <bool propto, bool jacobian_adjust_transforms>
      // double grad_log_prob(std::vector<double>& params_r,
      //                      std::vector<int>& params_i,
      //                      std::vector<double>& gradient,
      //                      std::ostream* output_stream = 0) { 
      //   return 0; 
      // }

      double log_prob(std::vector<double>& params_r,
                      std::vector<int>& params_i,
                      std::ostream* output_stream = 0) const { 
        return 0; 
      }
      
      
    };

    // Mock Hamiltonian
    template <typename M, typename BaseRNG>
    class mock_hamiltonian: public base_hamiltonian<M,
                                                    ps_point,
                                                    BaseRNG> {
      
    public:
      
      mock_hamiltonian(M& m, std::ostream *e): base_hamiltonian<M,
                                               ps_point,
                                               BaseRNG> (m,e) {};
      
      double T(ps_point& z) { return 0; }
      
      double tau(ps_point& z) { return T(z); }
      double phi(ps_point& z) { return this->V(z); }
      
      const Eigen::VectorXd dtau_dq(ps_point& z) {
        return Eigen::VectorXd::Zero(this->_model.num_params_r());
      }
      
      const Eigen::VectorXd dtau_dp(ps_point& z) {
        return Eigen::VectorXd::Zero(this->_model.num_params_r());
      }
      
      const Eigen::VectorXd dphi_dq(ps_point& z) {
        return Eigen::VectorXd::Zero(this->_model.num_params_r());
      }
      
      void sample_p(ps_point& z, BaseRNG& rng) {};
      
    };
    
    // Mock Integrator
    template <typename H, typename P>
    class mock_integrator: public base_integrator<H, P> {
    public:
      mock_integrator(std::ostream* o) 
      : base_integrator<H,P>(o)
      { }
      
      void evolve(P& z, H& hamiltonian, const double epsilon) {};
    };
    
  } // mcmc
  
} // stan

#endif
