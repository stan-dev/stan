#ifndef __STAN__MCMC__BASE__METRO__HPP__
#define __STAN__MCMC__BASE__METRO__HPP__

#include <iostream>
#include <vector>
#include <algorithm>

#include <stan/mcmc/base_mcmc.hpp>
#include <stan/prob/distributions/univariate/continuous/normal.hpp>
#include <stan/prob/distributions/univariate/continuous/uniform.hpp>

namespace stan {
  
  namespace mcmc {
    
    // Metropolis-Hastings Algorithm
        
    template <class M, class BaseRNG>
    class base_metro: public base_mcmc {
      
    public:
      
      base_metro(M& m, BaseRNG& rng, std::ostream* o, std::ostream* e)
        : base_mcmc(o,e), 
          _model(m),
          _params_r(m.num_params_r()),
          _params_i(m.num_params_i()),
          _rand_int(rng),
          _rand_uniform(_rand_int),
          _nom_epsilon(0.1),
          _epsilon(_nom_epsilon),
          _epsilon_jitter(0.0) {};  
 
      ~base_metro() {};

      virtual void propose(std::vector<double>& q, BaseRNG& rng) = 0;

      virtual void write_metric(std::ostream& o) = 0;

      void seed(const std::vector<double>& q, const std::vector<int>& r) {
        _params_r = q;
        _params_i = r;
      }     

     sample transition(sample& init_sample) {

       this->_sample_stepsize();

       this->seed(init_sample.cont_params(), init_sample.disc_params());

       std::vector<double> init_point(_params_r.size());
       init_point = _params_r;

       double logp0 = log_prob(_params_r, _params_i);

       this->propose(_params_r, _rand_int);

       double accept_prob = exp(log_prob(_params_r, _params_i) 
                                - logp0);

       double accept = true;
       if (accept_prob < 1 && this->_rand_uniform() > accept_prob) {
         _params_r = init_point;
         accept = false;
       }

       accept_prob = accept_prob > 1 ? 1 : accept_prob;

       return sample(_params_r, 
                     _params_i,
                     log_prob(_params_r, _params_i),
                     accept_prob);
     }

      double log_prob(std::vector<double>& q, std::vector<int>& r) {
        return _model.stan::model::prob_grad_ad::log_prob(q, r, this->_err_stream);
      }

      void init_stepsize() {
        std::vector<double> params_r0(_params_r);
        std::vector<int> params_i0(_params_i);

        this->propose(_params_r, _rand_int);
        double log_p0 = log_prob(_params_r, _params_i);

        this->propose(_params_r, _rand_int);        

        double log_p = log_prob(_params_r, _params_i);
        double delta_log_p = log_p0 - log_p;

        int direction = delta_log_p > log(0.5) ? 1 : -1;
        
        while (1) {                  
          this->seed(params_r0, params_i0);

          this->propose(_params_r, _rand_int);
          double log_p0 = log_prob(_params_r, _params_i);

          this->propose(_params_r, _rand_int);        

          double log_p = log_prob(_params_r, _params_i);
          double delta_log_p = log_p0 - log_p;
                
          if ((direction == 1) && !(delta_log_p > log(0.5)))
            break;
          else if ((direction == -1) && !(delta_log_p < log(0.5)))
            break;
          else
            this->_nom_epsilon = ( (direction == 1)
                                   ? 2.0 * this->_nom_epsilon
                                   : 0.5 * this->_nom_epsilon );
          
          if (this->_nom_epsilon > 1e300)
            throw std::runtime_error("Posterior is improper. Please check your model.");
          if (this->_nom_epsilon == 0)
            throw std::runtime_error("No acceptably small step size could be found. Perhaps the posterior is not continuous?");
        }        
      }      
      
      void write_sampler_param_names(std::ostream& o) {
        o << "stepsize__,";
      }
      
      void write_sampler_params(std::ostream& o) {
        o << this->_epsilon << ",";
      }
      
      void get_sampler_param_names(std::vector<std::string>& names) {
        names.clear();
        names.push_back("stepsize__");
      }
      
      void get_sampler_params(std::vector<double>& values) {
        values.clear();
        values.push_back(this->_epsilon);
      }
      
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
      M _model;

      std::vector<double> _params_r;
      std::vector<int> _params_i;

      BaseRNG& _rand_int;
      boost::uniform_01<BaseRNG&> _rand_uniform;                

      double _nom_epsilon;
      double _epsilon;
      double _epsilon_jitter;
      std::ostream* _error_msg;

      void _write_error_msg(std::ostream* error_msgs,
                           const std::domain_error& e) {
    }

      void _sample_stepsize() {
        this->_epsilon = this->_nom_epsilon;
        if(this->_epsilon_jitter)
          this->_epsilon *= ( 1.0 + this->_epsilon_jitter 
                                      * (2.0 * this->_rand_uniform() - 1.0) );
      }
    };
    
  } // mcmc
  
} // stan


#endif
