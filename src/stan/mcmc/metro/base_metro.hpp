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
          _rand_int(rng),
          _rand_uniform(_rand_int),
          _nom_epsilon(0.1),
          _epsilon(_nom_epsilon),
          _epsilon_jitter(0.0) {};  
 
      ~base_metro() {};

      virtual void propose(Eigen::VectorXd& q, BaseRNG& rng) = 0;
      virtual void write_metric(std::ostream* o) = 0;

      void write_sampler_state(std::ostream* o) {
        if(!o) return;
        *o << "# Step size = " << get_nominal_stepsize() << std::endl;
        this->write_metric(o);
      }
      
      void get_sampler_diagnostic_names(std::vector<std::string>& model_names,
                                        std::vector<std::string>& names) {
        this->get_param_names(model_names, names);
      };
      
      void get_sampler_diagnostics(std::vector<double>& values) {
        this->get_params(values);
      };

      void seed(const Eigen::VectorXd& q) {
        _params_r = q;
      }     

     sample transition(sample& init_sample) {

       this->_sample_stepsize();

       this->seed(init_sample.cont_params(), init_sample.disc_params());

       double logp0 = log_prob(_params_r, _params_i);

       this->propose(_params_r, _rand_int);
       double log_p = log_prob(_params_r,_params_i);

       if (boost::math::isnan(log_p)) 
         log_p = std::numeric_limits<double>::infinity();

       double accept_prob = std::exp(log_p - logp0);
       accept_prob = accept_prob > 1 ? 1 : accept_prob;

       if (this->_rand_uniform() > accept_prob) {
         _params_r = init_sample.cont_params();
         log_p = logp0;
       }

       return sample(_params_r, 
                     _params_i,
                     log_p,
                     accept_prob);
     }

      double log_prob(Eigen::VectorXd& q) {
        try {
          _model.template log_prob<false,true>(q, this->_err_stream);
        } catch (std::domain_error e) {
          this->_write_error_msg(this->_err_stream, e);
          return std::numeric_limits<double>::infinity();
        }
        return _model.template log_prob<false,true>(q, this->_err_stream);
      }

      void init_stepsize() {
        Eigen::VectorXd params_r0(this->_params_r);

        double log_p0 = log_prob(this->_params_r);

        this->propose(this->_params_r, _rand_int); 
        double log_p = log_prob(this->_params_r);

        if (boost::math::isnan(log_p)) 
          log_p = std::numeric_limits<double>::infinity();

        double delta_log_p = log_p - log_p0;

        int direction = delta_log_p > std::log(0.5) ? 1 : -1;
        while (1) {    
          this->seed(params_r0, params_i0);
          
          this->propose(this->_params_r, _rand_int); 
          log_p = log_prob(this->_params_r);
               
          delta_log_p = log_p - log_p0;

          if ((direction == 1) && !(delta_log_p > std::log(0.5)))
            break;
          else if ((direction == -1) && !(delta_log_p < std::log(0.5)))
            break;
          else
            this->_nom_epsilon = ( (direction == 1)
                                   ? 2.0 * this->_nom_epsilon
                                   : 0.5 * this->_nom_epsilon);
          
          if (this->_nom_epsilon > 1e7)
            throw std::runtime_error("Posterior is improper. Please check your model.");
          if (this->_nom_epsilon == 0)
            throw std::runtime_error("No acceptably small step size could be found. Perhaps the posterior is not continuous?");
        }        
      }      

      void get_params(std::vector<double>& values) {
        for(size_t i = 0; i < _params_r.size(); ++i)
          values.push_back(_params_r.at(i));
       }
      
      void get_param_names(std::vector<std::string>& model_names,
                           std::vector<std::string>& names) {
        for(size_t i = 0; i < _params_r.size(); ++i)
          names.push_back(model_names.at(i));
       }

      void write_sampler_param_names(std::ostream& o) {
        o << "stepsize__,";
      }
      
      void write_sampler_params(std::ostream& o) {
        o << this->_epsilon << ",";
      }
      
      void get_sampler_param_names(std::vector<std::string>& names) {
        names.push_back("stepsize__");
      }
      
      void get_sampler_params(std::vector<double>& values) {
        values.push_back(this->_epsilon);
      }
      
      void set_nominal_stepsize(const double e) {
        if(e > 0) _nom_epsilon = e;
      }
      
      double get_nominal_stepsize() { return this->_nom_epsilon; }
      
      double get_current_stepsize() { return this->_epsilon; }
      
      void set_stepsize_jitter(const double j) {
        if(j > 0 && j < 1) _epsilon_jitter = j;
      }
      
      double get_stepsize_jitter() { return this->_epsilon_jitter; }

      
    protected:
      M _model;

      std::vector<double> _params_r;

      BaseRNG& _rand_int;
      boost::uniform_01<BaseRNG&> _rand_uniform;                

      double _nom_epsilon;
      double _epsilon;
      double _epsilon_jitter;

      void _write_error_msg(std::ostream* error_msgs,
                           const std::domain_error& e) {
          if (!error_msgs) return;
          
          *error_msgs << std::endl
                      << "Informational Message: The parameter state is about to be Metropolis"
                      << " rejected due to the following underlying, non-fatal (really)"
                      << " issue (and please ignore that what comes next might say 'error'): "
                      << e.what()
                      << std::endl
                      << "If the problem persists across multiple draws, you might have"
                      << " a problem with an initial state."
                      << std::endl
                      << " If the problem does not persist, the resulting samples will still"
                      << " be drawn from the posterior."
                      << std::endl;
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
