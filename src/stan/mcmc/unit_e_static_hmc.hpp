#ifndef __STAN__MCMC__UNIT__E_STATIC__HMC__BETA__
#define __STAN__MCMC__UNIT__E_STATIC__HMC__BETA__

#include <stan/mcmc/stepsize_adapter.hpp>
#include <stan/mcmc/base_hmc.hpp>
#include <stan/mcmc/unit_e_point.hpp>
#include <stan/mcmc/unit_e_metric.hpp>
#include <stan/mcmc/expl_leapfrog.hpp>

namespace stan {

  namespace mcmc {

    // Hamiltonian Monte Carlo on a 
    // Euclidean manifold with unit metric
    // and static integration time
    
    template <typename M, class BaseRNG>
    class unit_e_static_hmc: public base_hmc<M, 
                                           unit_e_point,
                                           unit_e_metric, 
                                           expl_leapfrog, 
                                           BaseRNG> {
      
    public:
      
      unit_e_static_hmc(M &m, BaseRNG& rng);
      ~unit_e_static_hmc() {};
      
      sample transition(sample& init_sample);
                                             
      void write_sampler_param_names(std::ostream& o);
      void write_sampler_params(std::ostream& o);
      void get_sampler_param_names(std::vector<std::string>& names);
      void get_sampler_params(std::vector<double>& values);
      
      void set_stepsize_and_T(const double e, const double t) {
        _epsilon = e; _T = t; _update_L();
      }
      
      void set_stepsize_and_L(const double e, const int l) {
        _epsilon = e; _L = l; _T = _epsilon * _L;
      }
      
      void set_T(const double t) { _T = t; _update_L(); }
      void set_stepsize(const double e) { _epsilon = e; _update_L(); }
      
      double get_stepsize() { return this->_epsilon; }
      double get_T() { return this->_T; }
      int get_L() { return this->_L; }
      
    protected:
      
      double _epsilon;
      double _T;
      int _L;
      
      void _update_L() { _L = static_cast<int>(_T / _epsilon); }
                        
    };
    
    template <typename M, class BaseRNG>
    unit_e_static_hmc<M, BaseRNG>::unit_e_static_hmc(M& m, BaseRNG& rng):
    base_hmc<M, unit_e_point, unit_e_metric, expl_leapfrog, BaseRNG>(m, rng),
    _epsilon(0.1),
    _T(1)
    { _update_L(); }
    
    template <typename M, class BaseRNG>
    sample unit_e_static_hmc<M, BaseRNG>::transition(sample& init_sample) {
      
      unit_e_point z(init_sample.size_cont(), init_sample.size_disc());
      z.q = init_sample.cont_params();
      z.r = init_sample.disc_params();
      
      Eigen::VectorXd u(init_sample.size_cont());
      for (size_t i = 0; i < u.size(); ++i) u(i) = this->_rand_unit_gaus();
      
      this->_hamiltonian.sampleP(z, u);
      this->_hamiltonian.init(z);
      
      double H0 = this->_hamiltonian.H(z);
      
      for (int i = 0; i < _L; ++i) {
        this->_integrator.evolve(z, this->_hamiltonian, _epsilon);
      }
      
      double acceptProb = exp(H0 - this->_hamiltonian.H(z));
      
      double accept = true;
      if (acceptProb < 1 && this->_rand_uniform() > acceptProb) {
        z.q = init_sample.cont_params();
        accept = false;
      }
      
      return sample(z.q, z.r, - this->_hamiltonian.V(z), acceptProb);
      
    }
    
    template <typename M, class BaseRNG>
    void unit_e_static_hmc<M, BaseRNG>::write_sampler_param_names(std::ostream& o) {
        o << "stepsize__,int_time__,";
    }
    
    template <typename M, class BaseRNG>
    void unit_e_static_hmc<M, BaseRNG>::write_sampler_params(std::ostream& o) {
        o << this->_epsilon << "," << this->_T << ",";
    }
    
    template <typename M, class BaseRNG>
    void unit_e_static_hmc<M, BaseRNG>::get_sampler_param_names(std::vector<std::string>& names) {
      names.clear();
      names.push_back("stepsize__");
      names.push_back("int_time__");
    }
    
    template <typename M, class BaseRNG>
    void unit_e_static_hmc<M, BaseRNG>::get_sampler_params(std::vector<double>& values) {
      values.clear();
      values.push_back(this->_epsilon);
      values.push_back(this->_T);
    }
    
  } // mcmc

} // stan
          

#endif
