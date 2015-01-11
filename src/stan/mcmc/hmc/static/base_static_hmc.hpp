#ifndef STAN__MCMC__BASE__STATIC__HMC__BETA
#define STAN__MCMC__BASE__STATIC__HMC__BETA

#include <math.h>
#include <boost/math/special_functions/fpclassify.hpp>
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
      base_hmc<M, P, H, I, BaseRNG>(m, rng, o, e), T_(1)
      { update_L_(); }
      
      ~base_static_hmc() {};
      
      sample transition(sample& init_sample) {
        
        this->sample_stepsize();
        
        this->seed(init_sample.cont_params());
        
        this->hamiltonian_.sample_p(this->z_, this->rand_int_);
        this->hamiltonian_.init(this->z_);
        
        ps_point z_init(this->z_);

        double H0 = this->hamiltonian_.H(this->z_);
        
        for (int i = 0; i < L_; ++i)
          this->integrator_.evolve(this->z_, this->hamiltonian_, this->epsilon_);
        
        double h = this->hamiltonian_.H(this->z_);
        if (boost::math::isnan(h)) h = std::numeric_limits<double>::infinity();
        
        double acceptProb = std::exp(H0 - h);
        
        if (acceptProb < 1 && this->rand_uniform_() > acceptProb)
          this->z_.ps_point::operator=(z_init);
        
        acceptProb = acceptProb > 1 ? 1 : acceptProb;
        
        return sample(this->z_.q, - this->hamiltonian_.V(this->z_), acceptProb);
        
      }
      
      void write_sampler_param_names(std::ostream& o) {
        o << "stepsize__,int_time__,";
      }
      
      void write_sampler_params(std::ostream& o) {
        o << this->epsilon_ << "," << this->T_ << ",";
      }
      
      void get_sampler_param_names(std::vector<std::string>& names) {
        names.push_back("stepsize__");
        names.push_back("int_time__");
      }
      
      void get_sampler_params(std::vector<double>& values) {
        values.push_back(this->epsilon_);
        values.push_back(this->T_);
      }
      
      void set_nominal_stepsize_and_T(const double e, const double t) {
        if(e > 0 && t > 0) {
          this->nom_epsilon_ = e;
          T_ = t;
          update_L_();
        }
      }
      
      void set_nominal_stepsize_and_L(const double e, const int l) {
        if(e > 0 && l > 0) {
          this->nom_epsilon_ = e;
          L_ = l;
          T_ = this->nom_epsilon_ * L_; }
      }
      
      void set_T(const double t) { 
        if(t > 0) {
          T_ = t;
          update_L_();
        }
      }
      
      void set_nominal_stepsize(const double e) {
        if(e > 0) {
          this->nom_epsilon_ = e;
          update_L_();
        }
      }
      
      double get_T() { return this->T_; }
      int get_L() { return this->L_; }
      
    protected:
      
      double T_;
      int L_;
      
      void update_L_() {
        L_ = static_cast<int>(T_ / this->nom_epsilon_);
        L_ = L_ < 1 ? 1 : L_;
      }
      
    };
    
  } // mcmc
  
} // stan


#endif
