#ifndef __STAN__MCMC__BASE__HMC__BETA__
#define __STAN__MCMC__BASE__HMC__BETA__

#include <stdexcept>

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
    
      base_hmc(M &m, BaseRNG& rng): base_mcmc(),
                                    _z(m.num_params_r(), m.num_params_i()),
                                    _hamiltonian(m), 
                                    _rand_int(rng),
                                    _rand_uniform(_rand_int),
                                    _epsilon(0.1)
      {};
      
      void seed(const std::vector<double>& q, const std::vector<int>& r) {
        _z.q = q;
        _z.r = r;
      }
      
      void init_stepsize() {
  
        ps_point z_init(static_cast<ps_point>(this->_z));
  
        this->_hamiltonian.sample_p(this->_z, this->_rand_int);
        this->_hamiltonian.init(this->_z);
        
        double H0 = this->_hamiltonian.H(this->_z);
        this->_integrator.evolve(this->_z, this->_hamiltonian, this->_epsilon);
        double delta_H = H0 - this->_hamiltonian.H(this->_z);
        
        int direction = delta_H > log(0.5) ? 1 : -1;
        
        while (1) {
          
          this->_z.copy_base(z_init);
          
          this->_hamiltonian.sample_p(this->_z, this->_rand_int);
          this->_hamiltonian.init(this->_z);
          
          double H0 = this->_hamiltonian.H(this->_z);
          this->_integrator.evolve(this->_z, this->_hamiltonian, this->_epsilon);
          double delta_H = H0 - this->_hamiltonian.H(this->_z);
                   
          if ((direction == 1) && !(delta_H > log(0.5))) 
            break;
          else if ((direction == -1) && !(delta_H < log(0.5)))
            break;
          else
            this->_epsilon = ( (direction == 1) 
                              ? 2.0 * this->_epsilon 
                              : 0.5 * this->_epsilon );
          
          if (this->_epsilon > 1e300)
            throw std::runtime_error("Posterior is improper. Please check your model.");
          if (this->_epsilon == 0)
            throw std::runtime_error("No acceptably small step size could be found. Perhaps the posterior is not continuous?");
          
        }
        
        this->_z.copy_base(z_init);
        
      }
      
      P& z() { return _z; }
      
      virtual void set_stepsize(const double e) { 
        if(e > 0) _epsilon = e;
      }
      
      double get_stepsize() { return this->_epsilon; }
      
    protected:
    
      P _z;
      I<H<M, BaseRNG>, P> _integrator;
      H<M, BaseRNG> _hamiltonian;
      
      BaseRNG& _rand_int;
      
      // Uniform(0, 1) RNG
      boost::uniform_01<BaseRNG&> _rand_uniform;                
      
      double _epsilon;
    
    };
    
  } // mcmc

} // stan

#endif
