#ifndef STAN_IO_MCMC_WRITER_HPP
#define STAN_IO_MCMC_WRITER_HPP

#include <stan/interface_callbacks/writer/csv.hpp>
#include <stan/mcmc/base_mcmc.hpp>
#include <stan/mcmc/sample.hpp>
#include <stan/model/prob_grad.hpp>
#include <ostream>
#include <iomanip>
#include <string>
#include <vector>

namespace stan {

  namespace io {

    /**
     * mcmc_writer writes out headers and samples
     *
     * @tparam M Model class
     * @tparam SampleWriter Class for recording samples
     * @tparam DiagnosticWriter Class for diagnostic samples
     */
    template <class M,
              class SampleWriter, class DiagnosticWriter,
              class MessageWriter>
    class mcmc_writer {
    private:
      SampleWriter& sample_writer_;
      DiagnosticWriter& diagnostic_writer_;
      MessageWriter& message_writer_;

      std::ostream* msg_stream_;

    public:
      /**
       * Constructor.
       *
       * @param sample_writer samples are "written" to this stream (can abstract this?)
       * @param diagnostic_writer diagnostic information is "written" to this stream
       * @param message_writer messages are written to this stream
       * @param msg_stream messages are output to this stream
       *
       * @pre arguments == 0 if and only if they are not meant to be used
       * @post none
       * @sideeffects streams are stored in this object
       */
      mcmc_writer(SampleWriter& sample_writer,
                  DiagnosticWriter& diagnostic_writer,
                  MessageWriter& message_writer,
                  std::ostream* msg_stream = 0)
        : sample_writer_(sample_writer),
          diagnostic_writer_(diagnostic_writer),
          message_writer_(message_writer),
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
      void write_sample_names(stan::mcmc::sample& sample,
                              stan::mcmc::base_mcmc* sampler,
                              M& model) {
        std::vector<std::string> names;

        sample.get_sample_param_names(names);
        sampler->get_sampler_param_names(names);
        model.constrained_param_names(names, true, true);

        sample_writer_(names);
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
      void write_sample_params(RNG& rng,
                               stan::mcmc::sample& sample,
                               stan::mcmc::base_mcmc& sampler,
                               M& model) {
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

        sample_writer_(values);
      }

      /**
       * Internal method
       *
       * Prints additional sampler info to the stream.
       *
       * @param sampler sampler
       * @param writer stream to output stuff to
       */
      template <class Writer>
      void write_adapt_finish(stan::mcmc::base_mcmc* sampler,
                              Writer& writer) {
        std::stringstream stream;
        sampler->write_sampler_state(&stream);

        writer("Adaptation terminated\n" + stream.str());
      }


      /**
       * Prints additional info to the streams
       *
       * Prints to both the sample stream and the diagnostic stream
       *
       * @param sampler sampler
       */
      void write_adapt_finish(stan::mcmc::base_mcmc* sampler) {
        write_adapt_finish(sampler, sample_writer_);
        write_adapt_finish(sampler, diagnostic_writer_);
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
      void write_diagnostic_names(stan::mcmc::sample sample,
                                  stan::mcmc::base_mcmc* sampler,
                                  M& model) {
        std::vector<std::string> names;

        sample.get_sample_param_names(names);
        sampler->get_sampler_param_names(names);

        std::vector<std::string> model_names;
        model.unconstrained_param_names(model_names, false, false);

        sampler->get_sampler_diagnostic_names(model_names, names);

        diagnostic_writer_(names);
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
      void write_diagnostic_params(stan::mcmc::sample& sample,
                                   stan::mcmc::base_mcmc* sampler) {
        std::vector<double> values;

        sample.get_sample_params(values);
        sampler->get_sampler_params(values);
        sampler->get_sampler_diagnostics(values);

        diagnostic_writer_(values);
      }


      /**
       * Internal method
       *
       * Prints timing information
       *
       * @param warmDeltaT warmup time in seconds
       * @param sampleDeltaT sample time in seconds
       * @param writer output stream
       *
       * @pre none
       * @post none
       * @sideeffects stream is updated with information about timing
       *
       */
      template <class Writer>
      void write_timing(double warmDeltaT, double sampleDeltaT,
                        Writer& writer) {
        std::string title(" Elapsed Time: ");
        std::stringstream ss;


        writer();

        ss.str("");
        ss << title << warmDeltaT << " seconds (Warm-up)";
        writer(ss.str());

        ss.str("");
        ss << std::string(title.size(), ' ') << sampleDeltaT
           << " seconds (Sampling)";
        writer(ss.str());

        ss.str("");
        ss << std::string(title.size(), ' ')
           << warmDeltaT + sampleDeltaT
           << " seconds (Total)";
        writer(ss.str());

        writer();
      }


      /**
       * Print timing information to all streams
       *
       * @param warmDeltaT warmup time (sec)
       * @param sampleDeltaT sample time (sec)
       */
      void write_timing(double warmDeltaT, double sampleDeltaT) {
        write_timing(warmDeltaT, sampleDeltaT, sample_writer_);
        write_timing(warmDeltaT, sampleDeltaT, diagnostic_writer_);
        write_timing(warmDeltaT, sampleDeltaT, message_writer_);
      }
    };

  }  // io
}  // stan

#endif
