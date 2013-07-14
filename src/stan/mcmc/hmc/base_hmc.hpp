#ifndef __STAN__MCMC__BASE__HMC__BETA__
#define __STAN__MCMC__BASE__HMC__BETA__

#include <math.h>
#include <stdexcept>

#include <boost/math/special_functions/fpclassify.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/uniform_01.hpp>

#include <stan/mcmc/base_mcmc.hpp>
#include <stan/mcmc/hmc/hamiltonians/ps_point.hpp>

namespace stan {

  namespace mcmc {

    template <class M, class P, template<class, class> class H, 
              template<class, class> class I, class BaseRNG>
    class base_hmc: public base_mcmc {
    
    public:
    
      base_hmc(M &m, BaseRNG& rng, std::ostream* o, std::ostream* e):
      base_mcmc(o, e),
      _z(m.num_params_r(), m.num_params_i()),
      _integrator(this->_out_stream),
      _hamiltonian(m, this->_err_stream),
      _rand_int(rng),
      _rand_uniform(_rand_int),
      _nom_epsilon(0.1),
      _epsilon(_nom_epsilon),
      _epsilon_jitter(0.0)
      {};
      
      void write_sampler_state(std::ostream* o) {
        if(!o) return;
        *o << "# Step size = " << get_nominal_stepsize() << std::endl;
        _z.write_metric(o);
      }
      
      void get_sampler_diagnostic_names(std::vector<std::string>& model_names,
                                        std::vector<std::string>& names) {
        _z.get_param_names(model_names, names);
      };
      
      void get_sampler_diagnostics(std::vector<double>& values) {
        _z.get_params(values);
      };
      
      void seed(const std::vector<double>& q, const std::vector<int>& r) {
        _z.q = q;
        _z.r = r;
      }
      
      void init_stepsize() {
  
        ps_point z_init(this->_z);
  
        this->_hamiltonian.sample_p(this->_z, this->_rand_int);
        this->_hamiltonian.init(this->_z);
        
        double H0 = this->_hamiltonian.H(this->_z); // Guaranteed to be finite if randomly initialized
        
        this->_integrator.evolve(this->_z, this->_hamiltonian, this->_nom_epsilon);
        
        double h = this->_hamiltonian.H(this->_z);
        if (boost::math::isnan(h)) h = std::numeric_limits<double>::infinity();
        
        double delta_H = H0 - h;
        
        int direction = delta_H > std::log(0.5) ? 1 : -1;
        
        while (1) {
          
          this->_z.ps_point::operator=(z_init);
          
          this->_hamiltonian.sample_p(this->_z, this->_rand_int);
          this->_hamiltonian.init(this->_z);
          
          double H0 = this->_hamiltonian.H(this->_z);
          
          this->_integrator.evolve(this->_z, this->_hamiltonian, this->_nom_epsilon);
          
          double h = this->_hamiltonian.H(this->_z);
          if (boost::math::isnan(h)) h = std::numeric_limits<double>::infinity();
          
          double delta_H = H0 - h;
          
          if ((direction == 1) && !(delta_H > std::log(0.5)))
            break;
          else if ((direction == -1) && !(delta_H < std::log(0.5)))
            break;
          else
            this->_nom_epsilon = ( (direction == 1)
                                   ? 2.0 * this->_nom_epsilon
                                   : 0.5 * this->_nom_epsilon );
          
          if (this->_nom_epsilon > 1e7)
            throw std::runtime_error("Posterior is improper. Please check your model.");
          if (this->_nom_epsilon == 0)
            throw std::runtime_error("No acceptably small step size could be found. Perhaps the posterior is not continuous?");
          
        }
        
        this->_z.ps_point::operator=(z_init);
        
      }
      
      P& z() { return _z; }

      virtual void set_nominal_stepsize(const double e) {
        if(e > 0) _nom_epsilon = e;
      }
      
      double get_nominal_stepsize() { return this->_nom_epsilon; }
      
      double get_current_stepsize() { return this->_epsilon; }
      
      virtual void set_stepsize_jitter(const double j) {
        if(j > 0 && j < 1) _epsilon_jitter = j;
      }
      
      double get_stepsize_jitter() { return this->_epsilon_jitter; }
      
    protected:
    
      P _z;
      I<H<M, BaseRNG>, P> _integrator;
      H<M, BaseRNG> _hamiltonian;
      
      BaseRNG& _rand_int;
      
      // Uniform(0, 1) RNG
      boost::uniform_01<BaseRNG&> _rand_uniform;                
      
      double _nom_epsilon;
      double _epsilon;
      double _epsilon_jitter;
      
      void _sample_stepsize() {
        this->_epsilon = this->_nom_epsilon;
        if(this->_epsilon_jitter)
          this->_epsilon *= ( 1.0 + this->_epsilon_jitter * (2.0 * this->_rand_uniform() - 1.0) );
      }
    
    };
    
  } // mcmc

} // stan

#endif
