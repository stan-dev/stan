#ifndef __STAN__MCMC__BASE__STATIC__HMC__BETA__
#define __STAN__MCMC__BASE__STATIC__HMC__BETA__

#include <math.h>
#include <stan/mcmc/hmc/base_hmc.hpp>
#include <stan/mcmc/hmc/hamiltonians/ps_point.hpp>

namespace stan {
  
  namespace mcmc {
    
    // Hamiltonian Monte Carlo
    // with static integration time
        
    template <class M, class P, template<class, class> class H, 
              template<class, class> class I, class BaseRNG>
    class base_static_hmc: public base_hmc<M, P, H, I, BaseRNG> {
      
    public:
      
      base_static_hmc(M &m, BaseRNG& rng, std::ostream* o, std::ostream* e):
      base_hmc<M, P, H, I, BaseRNG>(m, rng, o, e), _T(1)
      { _update_L(); }
      
      ~base_static_hmc() {};
      
      sample transition(sample& init_sample) {
        
        this->_sample_stepsize();
        
        this->seed(init_sample.cont_params(), init_sample.disc_params());
        
        this->_hamiltonian.sample_p(this->_z, this->_rand_int);
        this->_hamiltonian.init(this->_z);
        
        ps_point z_init(static_cast<ps_point>(this->_z));

        double H0 = this->_hamiltonian.H(this->_z);
        
        for (int i = 0; i < _L; ++i) {
          this->_integrator.evolve(this->_z, this->_hamiltonian, this->_epsilon);
        }
        
        double acceptProb = std::exp(H0 - this->_hamiltonian.H(this->_z));
        
        double accept = true;
        if (acceptProb < 1 && this->_rand_uniform() > acceptProb) {
          this->_z.copy_base(z_init);
          accept = false;
        }
        
        acceptProb = acceptProb > 1 ? 1 : acceptProb;
        
        return sample(this->_z.q, this->_z.r, - this->_hamiltonian.V(this->_z), acceptProb);
        
      }
      
      void write_sampler_param_names(std::ostream& o) {
        o << "stepsize__,int_time__,";
      }
      
      void write_sampler_params(std::ostream& o) {
        o << this->_epsilon << "," << this->_T << ",";
      }
      
      void get_sampler_param_names(std::vector<std::string>& names) {
        names.clear();
        names.push_back("stepsize__");
        names.push_back("int_time__");
      }
      
      void get_sampler_params(std::vector<double>& values) {
        values.clear();
        values.push_back(this->_epsilon);
        values.push_back(this->_T);
      }
      
      void set_nominal_stepsize_and_T(const double e, const double t) {
        if(e > 0 && t > 0) {
          this->_nom_epsilon = e; _T = t; _update_L();
        }
      }
      
      void set_nominal_stepsize_and_L(const double e, const int l) {
        if(e > 0 && l > 0) {
          this->_nom_epsilon = e; _L = l; _T = this->_nom_epsilon * _L;
        }
      }
      
      void set_T(const double t) { 
        if(t > 0) {
          _T = t; _update_L();
        }
        
      }
      
      void set_nominal_stepsize(const double e) {
        if(e > 0) {
          this->_nom_epsilon = e; _update_L();
        }
      }
      
      double get_T() { return this->_T; }
      int get_L() { return this->_L; }
      
    protected:
      
      double _T;
      int _L;
      
      void _update_L() { 
        _L = static_cast<int>(_T / this->_nom_epsilon);
        _L = _L < 1 ? 1 : _L;
      }
      
    };
    
  } // mcmc
  
} // stan


#endif
