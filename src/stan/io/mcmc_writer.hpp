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

    /**
     * Writes out a vector as string.
     */
    class as_csv {
    private:
      std::ostream *o_;
      const bool has_stream_;
      
    public:
      /**
       * Construct an object.
       *
       * @param o pointer to stream. Will accept 0.
       */
      as_csv(std::ostream *o) 
        : o_(o), has_stream_(o != 0) { }
      
      /**
       * Print vector as csv.
       *
       * Uses the insertion operator to write out the elements
       * as comma separated values, flushing the buffer after the
       * line is complete
       * 
       * @tparam T type of element
       * @param x vector of type T
       */
      template <class T>
      void operator()(const std::vector<T>& x) {
        if (!has_stream_)
          return;
        
        if (x.size() != 0) {
          *o_ << x.at(0);
          for (typename T::size_type n = 1; n < x.size(); n++) {
            *o_ << "," << x.at(n);
          }
        }
        *o_ << std::endl;
      }
      
      /**
       * Indicator function for whether the instance is recording.
       *
       * For this class, returns true if it has a stream.
       */
      bool is_recording() const {
        return has_stream_;
      }
    };

    
    /**
     * mcmc_writer writes out headers and samples
     *
     * @tparam M Model class
     */
    template <class M>
    class mcmc_writer {
      
    public:
      
      /**
       * Constructor.
       *
       * @param sample_stream samples are "written" to this stream (can abstract this?)
       * @param diagnostic_stream diagnostic information is "written" to this stream
       * @param msg_stream messages are output to this stream
       *
       * @pre arguments == 0 if and only if they are not meant to be used
       * @post none
       * @sideeffects streams are stored in this object 
       */
      mcmc_writer(std::ostream* sample_stream,
                  std::ostream* diagnostic_stream,
                  std::ostream* msg_stream = 0)
        : sample_stream_(sample_stream),
          diagnostic_stream_(diagnostic_stream),
          msg_stream_(msg_stream) {
      }
      
      /**
       * Outputs parameter string names. First outputs the names stored in 
       * the sample object (stan::mcmc::sample), then uses the sampler provided
       * to output sampler specific names, then adds the model constrained
       * parameter names.
       * 
       * The names are written to the sample_stream as comma separated values
       * with a newline at the end.
       *
       * @param sample a sample (unconstrained) that works with the model
       * @param sampler a stan::mcmc::base_mcmc object
       * @param model the model
       *
       * @pre none
       * @post none
       * @sideeffects sample_stream_ is written to with comma separated values
       *   with a newline at the end
       */
      void print_sample_names(stan::mcmc::sample& sample,
                              stan::mcmc::base_mcmc* sampler,
                              M& model) {
        if (!sample_stream_) 
          return;
        
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


      /**
       * Outputs samples. First outputs the values of the sample params
       * from a stan::mcmc::sample, then outputs the values of the sampler
       * params from a stan::mcmc::base_mcmc, then finally outputs the values
       * of the model.
       *
       * The samples are written to the sample_stream as comma separated values
       * with a newline at the end.
       *
       * @param rng random number generator (used by model.write_array())
       * @param sample the sample in constrained space
       * @param sampler the sampler
       * @param model the model
       */
      template <class RNG>
      void print_sample_params(RNG& rng, 
                               stan::mcmc::sample& sample,
                               stan::mcmc::base_mcmc& sampler,
                               M& model) {
        if (!sample_stream_) 
          return;
        
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
      
      /**
       * Prints additional sampler info to the stream.
       * 
       * @param sampler sampler
       * @param stream stream to output stuff to
       */
      void print_adapt_finish(stan::mcmc::base_mcmc* sampler, std::ostream* stream) {
        if (!stream) 
          return;
        
        *stream << "# Adaptation terminated" << std::endl;
        sampler->write_sampler_state(stream);
        
      }
      

      /**
       * Prints additional info to the streams
       *
       * Prints to both the sample stream and the diagnostic stream
       *
       * @param sampler sampler
       */
      void print_adapt_finish(stan::mcmc::base_mcmc* sampler) {
        print_adapt_finish(sampler, sample_stream_);
        print_adapt_finish(sampler, diagnostic_stream_);
      }


      /**
       * Print diagnostic names
       *
       * @param sample unconstrained sample
       * @param sampler sampler
       * @param model model
       *
       * @pre sample, sampler, and model are consistent.
       * @post none
       * @sideeffects diagnostic_stream_ is appended with comma
       *   separated names with newline at the end
       */
      void print_diagnostic_names(stan::mcmc::sample sample,
                                  stan::mcmc::base_mcmc* sampler,
                                  M& model) {
        if (!diagnostic_stream_) 
          return;
        
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
      
      /**
       * Print diagnostic params to the diagnostic stream.
       *
       * @param sample unconstrained sample
       * @param sampler sampler
       * 
       * @pre sample and sampler are consistent
       * @post none.
       * @sideeffects diagnostic_stream_ is appended with csv values of the
       *   sample's get_sample_params(), the sampler's get_sampler_params(),
       *   and get_sampler_diagnostics()
       */
      void print_diagnostic_params(stan::mcmc::sample& sample,
                                   stan::mcmc::base_mcmc* sampler) {
        if (!diagnostic_stream_) 
          return;
        
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
      
      /**
       * Prints timing information
       *
       * @param warmDeltaT warmup time in seconds
       * @param sampleDeltaT sample time in seconds
       * @param stream output stream
       * @param prefix prepend each line with the prefix; default is ""
       *
       * @pre none
       * @post none
       * @sideeffects stream is updated with information about timing
       *
       */
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
      
      
      /**
       * Print timing information to all streams
       *
       * @param warmDeltaT warmup time (sec)
       * @param sampleDeltaT sample time (sec)
       */
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
