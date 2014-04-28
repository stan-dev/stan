#ifndef __STAN__IO__MCMC__WRITER__HPP
#define __STAN__IO__MCMC__WRITER__HPP

#include <ostream>
#include <iomanip>
#include <string>
#include <vector>

#include <stan/mcmc/base_mcmc.hpp>
#include <stan/mcmc/sample.hpp>
#include <stan/model/prob_grad.hpp>

namespace stan {
  
  namespace io {
    
    template <class M>
    class mcmc_writer {
      
    public:
      
      mcmc_writer(std::ostream* sample_stream,
                  std::ostream* diagnostic_stream,
                  std::ostream* msg_stream = 0)
        : sample_stream_(sample_stream),
          diagnostic_stream_(diagnostic_stream),
          msg_stream_(msg_stream) {
      }
      
      void print_sample_names(stan::mcmc::sample& sample,
                              stan::mcmc::base_mcmc* sampler,
                              M& model) {
        
        if (!sample_stream_) return;
        
        std::vector<std::string> names;
        
        sample.get_sample_param_names(names);
        sampler->get_sampler_param_names(names);
        model.constrained_param_names(names, true, true);
       
        (*sample_stream_) << names.at(0);
        for (size_t i = 1; i < names.size(); ++i) {
          (*sample_stream_) << "," << names.at(i);
        }
        (*sample_stream_) << std::endl;
        
      }

      template <class RNG>
      void print_sample_params(RNG& rng, 
                               stan::mcmc::sample& sample,
                               stan::mcmc::base_mcmc& sampler,
                               M& model) {
        
        if (!sample_stream_) return;
        
        std::vector<double> values;
        
        sample.get_sample_params(values);
        sampler.get_sampler_params(values);
        
        Eigen::VectorXd model_values;
        
        model.write_array(rng,
                          const_cast<Eigen::VectorXd&>(sample.cont_params()),
                          model_values,
                          true, true,
                          msg_stream_); 

        for (int i = 0; i < model_values.size(); ++i)
          values.push_back(model_values(i));
        
        (*sample_stream_) << values.at(0);
        for (size_t i = 1; i < values.size(); ++i) {
          (*sample_stream_) << "," << values.at(i);
        }
        (*sample_stream_) << std::endl;
        
      }
      
      void print_adapt_finish(stan::mcmc::base_mcmc* sampler, std::ostream* stream) {
        
        if (!stream) return;
        
        *stream << "# Adaptation terminated" << std::endl;
        sampler->write_sampler_state(stream);
        
      }
      
      void print_adapt_finish(stan::mcmc::base_mcmc* sampler) {
        print_adapt_finish(sampler, sample_stream_);
        print_adapt_finish(sampler, diagnostic_stream_);
      }
      
      void print_diagnostic_names(stan::mcmc::sample sample,
                                  stan::mcmc::base_mcmc* sampler,
                                  M& model) {
        
        if (!diagnostic_stream_) return;
        
        std::vector<std::string> names;
        
        sample.get_sample_param_names(names);
        sampler->get_sampler_param_names(names);
        
        std::vector<std::string> model_names;
        model.unconstrained_param_names(model_names, false, false);
        
        sampler->get_sampler_diagnostic_names(model_names, names);
        
        for (size_t i = 0; i < names.size(); ++i) {
          if (i > 0) *diagnostic_stream_ << ",";
          *diagnostic_stream_ << names.at(i);
        }
        *diagnostic_stream_ << std::endl;
        
      }
      
      void print_diagnostic_params(stan::mcmc::sample& sample,
                                   stan::mcmc::base_mcmc* sampler) {
        
        if (!diagnostic_stream_) return;
        
        std::vector<double> values;
        
        sample.get_sample_params(values);
        sampler->get_sampler_params(values);
        sampler->get_sampler_diagnostics(values);
        
        (*diagnostic_stream_) << values.at(0);
        for (size_t i = 1; i < values.size(); ++i) {
          (*diagnostic_stream_) << "," << values.at(i);
        }
        (*diagnostic_stream_) << std::endl;
        
      }
      
      void print_timing(double warmDeltaT, double sampleDeltaT, 
                        std::ostream* stream, const std::string& prefix = "") {
        if (!stream) return;
        
        std::string title(" Elapsed Time: ");
        
        *stream << std::endl
                << prefix << " " << title << warmDeltaT
                << " seconds (Warm-up)"  << std::endl
                << prefix << " " << std::string(title.size(), ' ') << sampleDeltaT
                << " seconds (Sampling)"  << std::endl
                << prefix << " " << std::string(title.size(), ' ') 
                << warmDeltaT + sampleDeltaT
                << " seconds (Total)"  << std::endl
                << std::endl;
      }
      
      void print_timing(double warmDeltaT, double sampleDeltaT) {
        print_timing(warmDeltaT, sampleDeltaT, sample_stream_, "#");
        print_timing(warmDeltaT, sampleDeltaT, diagnostic_stream_, "#");
        print_timing(warmDeltaT, sampleDeltaT, &std::cout);
      }
      
    private:
      
      std::ostream* sample_stream_;
      std::ostream* diagnostic_stream_;
      std::ostream* msg_stream_;
      
    };
    
  } //io
  
} // stan

#endif
