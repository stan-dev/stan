#ifndef __STAN__IO__MCMC__WRITER__HPP
#define __STAN__IO__MCMC__WRITER__HPP

#include <iostream>
#include <iomanip>

#include <vector>
#include <string>

#include <stan/mcmc/sample.hpp>
#include <stan/mcmc/base_mcmc.hpp>
#include <stan/model/prob_grad.hpp>

namespace stan {
  
  namespace io {
    
    template <class M>
    class mcmc_writer {
      
    public:
      
      mcmc_writer(std::ostream* sample_stream,
                  std::ostream* diagnostic_stream): _sample_stream(sample_stream),
                                                    _diagnostic_stream(diagnostic_stream) {}
      
      void print_sample_names(stan::mcmc::sample& sample,
                              stan::mcmc::base_mcmc& sampler,
                              M& model) {
        
        if(!_sample_stream) return;
        
        std::vector<std::string> names;
        
        sample.get_sample_param_names(names);
        sampler.get_sampler_param_names(names);
        model.constrained_param_names(names, true, true);
        
        (*_sample_stream) << names.at(0);
        for (int i = 1; i < names.size(); ++i) {
          (*_sample_stream) << "," << names.at(i);
        }
        (*_sample_stream) << std::endl;
        
      }

      // need RNG in order to do random part of generated quantities
      template <class RNG>
      void print_sample_params(RNG& rng, 
                               stan::mcmc::sample& sample,
                               stan::mcmc::base_mcmc& sampler,
                               M& model) {
        
        if(!_sample_stream) return;
        
        std::vector<double> values;
        
        sample.get_sample_params(values);
        sampler.get_sampler_params(values);
        
        std::vector<double> model_values;
        
        model.write_array(rng,
                          const_cast<std::vector<double>&>(sample.cont_params()),
                          const_cast<std::vector<int>&>(sample.disc_params()),
                          model_values,
                          true, true); // FIXME: add ostream for msgs!
        
        values.insert(values.end(), model_values.begin(), model_values.end());
        
        (*_sample_stream) << values.at(0);
        for (int i = 1; i < values.size(); ++i) {
          (*_sample_stream) << "," << values.at(i);
        }
        (*_sample_stream) << std::endl;
        
      }
      
      void print_adapt_finish(stan::mcmc::base_mcmc& sampler, std::ostream* stream) {
        
        if(!stream) return;
        
        *stream << "# Adaptation terminated" << std::endl;
        sampler.write_sampler_state(stream);
        
      }
      
      void print_adapt_finish(stan::mcmc::base_mcmc& sampler) {
        print_adapt_finish(sampler, _sample_stream);
        print_adapt_finish(sampler, _diagnostic_stream);
      }
      
      void print_diagnostic_names(stan::mcmc::sample& sample,
                                  stan::mcmc::base_mcmc& sampler,
                                  M& model) {
        
        if(!_diagnostic_stream) return;
        
        std::vector<std::string> names;
        
        sample.get_sample_param_names(names);
        sampler.get_sampler_param_names(names);
        
        std::vector<std::string> model_names;
        model.unconstrained_param_names(model_names, false, false);
        
        sampler.get_sampler_diagnostic_names(model_names, names);
        
        (*_diagnostic_stream) << names.at(0);
        for (int i = 1; i < names.size(); ++i) {
          (*_diagnostic_stream) << "," << names.at(i);
        }
        (*_diagnostic_stream) << std::endl;
        
      }
      
      void print_diagnostic_params(stan::mcmc::sample& sample,
                                    stan::mcmc::base_mcmc& sampler) {
        
        if(!_diagnostic_stream) return;
        
        std::vector<double> values;
        
        sample.get_sample_params(values);
        sampler.get_sampler_params(values);
        sampler.get_sampler_diagnostics(values);
        
        (*_diagnostic_stream) << values.at(0);
        for (int i = 1; i < values.size(); ++i) {
          (*_diagnostic_stream) << "," << values.at(i);
        }
        (*_diagnostic_stream) << std::endl;
        
      }
      
      void print_timing(double warmDeltaT, double sampleDeltaT, std::ostream* stream) {
        if(!stream) return;
        
        std::string prefix("# Elapsed Time: ");
        
        *stream << std::endl
                << prefix << warmDeltaT
                << " seconds (Warm-up)"  << std::endl
                << "#" << std::string(prefix.size() - 1, ' ') << sampleDeltaT
                << " seconds (Sampling)"  << std::endl
                << "#" << std::string(prefix.size() - 1, ' ') << warmDeltaT + sampleDeltaT
                << " seconds (Total)"  << std::endl
                << std::endl;
      }
      
      void print_timing(double warmDeltaT, double sampleDeltaT) {
        print_timing(warmDeltaT, sampleDeltaT, _sample_stream);
        print_timing(warmDeltaT, sampleDeltaT, _diagnostic_stream);
        print_timing(warmDeltaT, sampleDeltaT, &std::cout);
      }
      
    private:
      
      std::ostream* _sample_stream;
      std::ostream* _diagnostic_stream;
      
    };
    
  } //io
  
} // stan

#endif
