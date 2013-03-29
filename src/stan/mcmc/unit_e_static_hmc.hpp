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
      
      unit_e_static_hmc(M &m, BaseRNG& rng): base_hmc<M, unit_e_point, unit_e_metric, expl_leapfrog, BaseRNG>(m, rng),
                                             _epsilon(0.1),
                                             _T(1)
      { _update_L(); }
                                               
      ~unit_e_static_hmc() {};
      
      sample transition(sample& init_sample) {
       
        unit_e_point z(init_sample.size_cont(), init_sample.size_disc());
        z.q = init_sample.cont_params();
        z.r = init_sample.disc_params();
       
        this->_hamiltonian.sample_p(z, this->_rand_int);
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
                                               
      
      void set_stepsize_and_T(const double e, const double t) {
        if(e > 0 && t > 0)
          _epsilon = e; _T = t; _update_L();
      }
      
      void set_stepsize_and_L(const double e, const int l) {
        if(e > 0 && l > 0)
          _epsilon = e; _L = l; _T = _epsilon * _L;
      }
      
      void set_T(const double t) { 
        if(t > 0)
          _T = t; _update_L(); 
      
      }
      void set_stepsize(const double e) { 
        if(e > 0)
            _epsilon = e; _update_L(); 
      }
      
      double get_stepsize() { return this->_epsilon; }
      double get_T() { return this->_T; }
      int get_L() { return this->_L; }
      
    protected:
      
      double _epsilon;
      double _T;
      int _L;
      
      void _update_L() { _L = static_cast<int>(_T / _epsilon); }
                        
    };

  } // mcmc

} // stan
          

#endif
