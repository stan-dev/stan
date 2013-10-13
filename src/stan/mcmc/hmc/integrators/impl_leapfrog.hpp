#ifndef __STAN__MCMC__IMPL__LEAPFROG__BETA__
#define __STAN__MCMC__IMPL__LEAPFROG__BETA__

#include <stan/math/matrix/Eigen.hpp>
#include <stan/mcmc/hmc/integrators/base_leapfrog.hpp>

namespace stan {
  
  namespace mcmc {
    
    template <typename H, typename P>
    class impl_leapfrog: public base_leapfrog<H, P> {
      
    public:
      
      impl_leapfrog(std::ostream* o=0): base_leapfrog<H, P>(o),
                                        _max_num_fixed_point(10),
                                        _fixed_point_threshold(1e-8) {};
      
      void begin_update_p(P& z, H& hamiltonian, double epsilon) {
        _hat_phi(z, hamiltonian, epsilon);
        _hat_tau(z, hamiltonian, epsilon, this->_max_num_fixed_point);
      }
      
      void update_q(P& z, H& hamiltonian, double epsilon) {
        _hat_T(z, hamiltonian, epsilon, this->_max_num_fixed_point);
      }
      
      void end_update_p(P& z, H& hamiltonian, double epsilon) {
        _hat_tau(z, hamiltonian, epsilon, 1);
        _hat_phi(z, hamiltonian, epsilon);
      }
      
      // hat{phi} = dphi/dq * d/dp
      void _hat_phi(P&z, H& hamiltonian, double epsilon) {
        z.p -= epsilon * hamiltonian.dphi_dq(z);
      }
      
      // hat{tau} = dtau/dq * d/dp
      void _hat_tau(P&z, H& hamiltonian, double epsilon, int num_fixed_point) {
        
        z.fp_init = z.p;
        
        for (int i = 0; i < num_fixed_point; ++i) {
          
          z.fp_delta.noalias() = z.p;
          z.p.noalias() = z.fp_init - epsilon * hamiltonian.dtau_dq(z);
          z.fp_delta -= z.p;
          
          hamiltonian.update_p(z);
          
          if(z.fp_delta.cwiseAbs().maxCoeff() < _fixed_point_threshold) break;
          
        }
        
      }
      
      // hat{T} = dT/dp * d/dq
      void _hat_T(P&z, H& hamiltonian, double epsilon, int num_fixed_point) {
        
        Eigen::Map<Eigen::VectorXd> q(&(z.q[0]), z.q.size());
        q += epsilon * hamiltonian.dtau_dp(z);
        
        z.fp_init.noalias() = q + 0.5 * epsilon * hamiltonian.dtau_dp(z);
        
        for (int i = 0; i < num_fixed_point; ++i) {
          
          z.fp_delta.noalias() = q;
          q.noalias() = z.fp_init + 0.5 * epsilon * hamiltonian.dtau_dp(z);
          z.fp_delta -= q;
          
          hamiltonian.compute_metric(z);
          
          if(z.fp_delta.cwiseAbs().maxCoeff() <_fixed_point_threshold) break;
          
        }
        
        hamiltonian.prepare_spatial_gradients(z);
        
      }
      
      int get_max_num_fixed_point() { return this->_max_num_fixed_point; }
      
      virtual void set_max_num_fixed_point(const int n) {
        if(n > 0) this->_max_num_fixed_point = n;
      }
      
      double get_fixed_point_threshold() { return this->_fixed_point_threshold; }
      
      virtual void set_fixed_point_threshold(const double t) {
        if(t > 0) this->_fixed_point_threshold = t;
      }

    private:
      
      size_t _max_num_fixed_point;
      double _fixed_point_threshold;
      
    };
    
  } // mcmc
  
} // stan


#endif
