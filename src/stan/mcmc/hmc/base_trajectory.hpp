#ifndef __STAN__MCMC__BASE__TRAJECTORY__BETA__
#define __STAN__MCMC__BASE__TRAJECTORY__BETA__

#include <stan/mcmc/hmc/base_hmc.hpp>
#include <stan/mcmc/hmc/hamiltonians/ps_point.hpp>

namespace stan {

  namespace mcmc {

    template <class M, class P, template<class, class> class H, 
              template<class, class> class I, class BaseRNG>
    class base_trajectory: public base_hmc<M, P, H, I, BaseRNG> {
    
    public:
        
      base_static_hmc(M &m, BaseRNG& rng, std::ostream* o, std::ostream* e):
      base_hmc<M, P, H, I, BaseRNG>(m, rng, o, e), T_(1)
      { _update_L(); }
      
      ~base_trajectory() {};
      
      void init(sample& init_sample) {
        
        this->seed(init_sample.cont_params(), init_sample.disc_params());
        
        this->_hamiltonian.sample_p(this->_z, this->_rand_int);
        this->_hamiltonian.init(this->_z);
        
      }
      
      void increment() {
        this->_integrator.evolve(this->_z, this->_hamiltonian, this->_epsilon);
      }
      
      sample transition(sample& init_sample) {
        return sample(this->_z.q, this->_z.r, - this->_hamiltonian.V(this->_z), acceptProb);
      }
      
      void write_sampler_param_names(std::ostream& o) {
        o << "stepsize__,int_time__,H__,T__,V__";
      }
      
      void write_sampler_params(std::ostream& o) {
        o << this->_epsilon << ","
          << this->T_ << ","
          << this->_hamiltonian.H(this->_z); << ","
          << this->_hamiltonian.T(this->_z); << ","
          << this->_hamiltonian.V(this->_z); << ",";
      }
      
      void get_sampler_param_names(std::vector<std::string>& names) {
        names.push_back("stepsize__");
        names.push_back("int_time__");
        names.push_back("H__");
        names.push_back("T__");
        names.push_back("V__");
      }
      
      void get_sampler_params(std::vector<double>& values) {
        values.push_back(this->_epsilon);
        values.push_back(this->T_);
        values.push_back(this->_hamiltonian.H(this->_z));
        values.push_back(this->_hamiltonian.T(this->_z));
        values.push_back(this->_hamiltonian.V(this->_z));
      }
      
      void set_nominal_stepsize_and_T(const double e, const double t) {
        if(e > 0 && t > 0) {
          this->_nom_epsilon = e; T_ = t; _update_L();
        }
      }
      
      void set_nominal_stepsize_and_L(const double e, const int l) {
        if(e > 0 && l > 0) {
          this->_nom_epsilon = e; L_ = l; T_ = this->_nom_epsilon * L_;
        }
      }
      
      void set_T(const double t) {
        if(t > 0) {
          T_ = t; _update_L();
        }
        
      }
      
      void set_nominal_stepsize(const double e) {
        if(e > 0) {
          this->_nom_epsilon = e; _update_L();
        }
      }
      
      double get_T() { return this->T_; }
      int get_L() { return this->L_; }
      
    protected:
      
      double T_;
      int L_;
      
      void _update_L() { 
        L_ = static_cast<int>(T_ / this->_nom_epsilon);
        L_ = L_ < 1 ? 1 : L_;
      }
      
    };
      
  } // mcmc

} // stan

#endif
