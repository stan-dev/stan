#ifndef STAN__MCMC__BASE__HMC__BETA
#define STAN__MCMC__BASE__HMC__BETA

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
      z_(m.num_params_r()),
      integrator_(this->out_stream_),
      hamiltonian_(m, this->err_stream_),
      rand_int_(rng),
      rand_uniform_(rand_int_),
      nom_epsilon_(0.1),
      epsilon_(nom_epsilon_),
      epsilon_jitter_(0.0)
      {};
      
      void write_sampler_state(std::ostream* o) {
        if(!o) return;
        *o << "# Step size = " << get_nominal_stepsize() << std::endl;
        z_.write_metric(o);
      }
      
      void get_sampler_diagnostic_names(std::vector<std::string>& model_names,
                                        std::vector<std::string>& names) {
        z_.get_param_names(model_names, names);
      };
      
      void get_sampler_diagnostics(std::vector<double>& values) {
        z_.get_params(values);
      };
      
      void seed(const Eigen::VectorXd& q) {
        z_.q = q;
      }
      
      void init_stepsize() {
  
        ps_point z_init(this->z_);
  
        this->hamiltonian_.sample_p(this->z_, this->rand_int_);
        this->hamiltonian_.init(this->z_);
        
        double H0 = this->hamiltonian_.H(this->z_); // Guaranteed to be finite if randomly initialized
        
        this->integrator_.evolve(this->z_, this->hamiltonian_, this->nom_epsilon_);
        
        double h = this->hamiltonian_.H(this->z_);
        if (boost::math::isnan(h)) h = std::numeric_limits<double>::infinity();
        
        double delta_H = H0 - h;
        
        int direction = delta_H > std::log(0.8) ? 1 : -1;
        
        while (1) {
          
          this->z_.ps_point::operator=(z_init);
          
          this->hamiltonian_.sample_p(this->z_, this->rand_int_);
          this->hamiltonian_.init(this->z_);
          
          double H0 = this->hamiltonian_.H(this->z_);
          
          this->integrator_.evolve(this->z_, this->hamiltonian_, this->nom_epsilon_);
          
          double h = this->hamiltonian_.H(this->z_);
          if (boost::math::isnan(h)) h = std::numeric_limits<double>::infinity();
          
          double delta_H = H0 - h;
          
          if ((direction == 1) && !(delta_H > std::log(0.8)))
            break;
          else if ((direction == -1) && !(delta_H < std::log(0.8)))
            break;
          else
            this->nom_epsilon_ = ( (direction == 1)
                                   ? 2.0 * this->nom_epsilon_
                                   : 0.5 * this->nom_epsilon_ );
          
          if (this->nom_epsilon_ > 1e7)
            throw std::runtime_error("Posterior is improper. Please check your model.");
          if (this->nom_epsilon_ == 0)
            throw std::runtime_error("No acceptably small step size could be found. Perhaps the posterior is not continuous?");
          
        }
        
        this->z_.ps_point::operator=(z_init);
        
      }
      
      P& z() { return z_; }

      virtual void set_nominal_stepsize(const double e) {
        if(e > 0) nom_epsilon_ = e;
      }
      
      double get_nominal_stepsize() { return this->nom_epsilon_; }
      
      double get_current_stepsize() { return this->epsilon_; }
      
      virtual void set_stepsize_jitter(const double j) {
        if(j > 0 && j < 1) epsilon_jitter_ = j;
      }
      
      double get_stepsize_jitter() { return this->epsilon_jitter_; }
      
      void sample_stepsize() {
        this->epsilon_ = this->nom_epsilon_;
        if(this->epsilon_jitter_)
          this->epsilon_ *= ( 1.0 + this->epsilon_jitter_ * (2.0 * this->rand_uniform_() - 1.0) );
      }
      
    protected:
    
      P z_;
      I<H<M, BaseRNG>, P> integrator_;
      H<M, BaseRNG> hamiltonian_;
      
      BaseRNG& rand_int_;
      
      // Uniform(0, 1) RNG
      boost::uniform_01<BaseRNG&> rand_uniform_;                
      
      double nom_epsilon_;
      double epsilon_;
      double epsilon_jitter_;
    
    };
    
  } // mcmc

} // stan

#endif
