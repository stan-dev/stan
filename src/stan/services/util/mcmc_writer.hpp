#ifndef STAN_SERVICES_UTIL_MCMC_WRITER_HPP
#define STAN_SERVICES_UTIL_MCMC_WRITER_HPP

#include <stan/mcmc/base_mcmc.hpp>
#include <stan/mcmc/sample.hpp>
#include <stan/model/prob_grad.hpp>
#include <sstream>
#include <iomanip>
#include <string>
#include <vector>

namespace stan {
  namespace services {
    namespace util {

      /**
       * mcmc_writer writes out headers and samples
       *
       * @tparam Model Model class
       */
      class mcmc_writer {
      private:
        callbacks::writer& sample_writer_;
        callbacks::writer& diagnostic_writer_;
        callbacks::writer& message_writer_;

      public:
        /**
         * Constructor.
         *
         * @param[in,out] sample_writer samples are "written" to this stream
         * @param[in,out] diagnostic_writer diagnostic info is "written" to this
         *   stream
         * @param[in,out] message_writer messages are written to this stream
         */
        mcmc_writer(callbacks::writer& sample_writer,
                    callbacks::writer& diagnostic_writer,
                    callbacks::writer& message_writer)
          : sample_writer_(sample_writer),
            diagnostic_writer_(diagnostic_writer),
            message_writer_(message_writer) {
        }

        /**
         * Outputs parameter string names. First outputs the names stored in
         * the sample object (stan::mcmc::sample), then uses the sampler
         * provided to output sampler specific names, then adds the model
         * constrained parameter names.
         *
         * The names are written to the sample_stream as comma separated values
         * with a newline at the end.
         *
         * @param[in] sample a sample (unconstrained) that works with the model
         * @param[in] sampler a stan::mcmc::base_mcmc object
         * @param[in] model the model
         */
        template <class Model>
        void write_sample_names(stan::mcmc::sample& sample,
                                stan::mcmc::base_mcmc& sampler,
                                Model& model) {
          std::vector<std::string> names;

          sample.get_sample_param_names(names);
          sampler.get_sampler_param_names(names);
          model.constrained_param_names(names, true, true);

          sample_writer_(names);
        }

        /**
         * Outputs samples. First outputs the values of the sample params
         * from a stan::mcmc::sample, then outputs the values of the sampler
         * params from a stan::mcmc::base_mcmc, then finally outputs the values
         * of the model.
         *
         * The samples are written to the sample_stream as comma separated
         * values with a newline at the end.
         *
         * @param[in,out] rng random number generator (used by
         *   model.write_array())
         * @param[in] sample the sample in constrained space
         * @param[in] sampler the sampler
         * @param[in] model the model
         */
        template <class Model, class RNG>
        void write_sample_params(RNG& rng,
                                 stan::mcmc::sample& sample,
                                 stan::mcmc::base_mcmc& sampler,
                                 Model& model) {
          std::vector<double> values;

          sample.get_sample_params(values);
          sampler.get_sampler_params(values);

          Eigen::VectorXd model_values;

          std::stringstream ss;
          model.write_array(rng,
                            const_cast<Eigen::VectorXd&>(sample.cont_params()),
                            model_values,
                            true, true,
                            &ss);
          if (ss.str().length() > 0)
            message_writer_(ss.str());

          for (int i = 0; i < model_values.size(); ++i)
            values.push_back(model_values(i));

          sample_writer_(values);
        }

        /**
         * Prints additional info to the streams
         *
         * Prints to the sample stream
         *
         * @param[in] sampler sampler
         */
        void write_adapt_finish(stan::mcmc::base_mcmc& sampler) {
          sample_writer_("Adaptation terminated");
          sampler.write_sampler_state(sample_writer_);
        }

        /**
         * Print diagnostic names
         *
         * @param[in] sample unconstrained sample
         * @param[in] sampler sampler
         * @param[in] model model
         */
        template <class Model>
        void write_diagnostic_names(stan::mcmc::sample sample,
                                    stan::mcmc::base_mcmc& sampler,
                                    Model& model) {
          std::vector<std::string> names;

          sample.get_sample_param_names(names);
          sampler.get_sampler_param_names(names);

          std::vector<std::string> model_names;
          model.unconstrained_param_names(model_names, false, false);

          sampler.get_sampler_diagnostic_names(model_names, names);

          diagnostic_writer_(names);
        }

        /**
         * Print diagnostic params to the diagnostic stream.
         *
         * @param[in] sample unconstrained sample
         * @param[in] sampler sampler
         */
        void write_diagnostic_params(stan::mcmc::sample& sample,
                                     stan::mcmc::base_mcmc& sampler) {
          std::vector<double> values;

          sample.get_sample_params(values);
          sampler.get_sampler_params(values);
          sampler.get_sampler_diagnostics(values);

          diagnostic_writer_(values);
        }

        /**
         * Internal method
         *
         * Prints timing information
         *
         * @param[in] warmDeltaT warmup time in seconds
         * @param[in] sampleDeltaT sample time in seconds
         * @param[in,out] writer output stream
         */
        void write_timing(double warmDeltaT, double sampleDeltaT,
                          callbacks::writer& writer) {
          std::string title(" Elapsed Time: ");
          writer();

          std::stringstream ss1;
          ss1 << title << warmDeltaT << " seconds (Warm-up)";
          writer(ss1.str());

          std::stringstream ss2;
          ss2 << std::string(title.size(), ' ') << sampleDeltaT
              << " seconds (Sampling)";
          writer(ss2.str());

          std::stringstream ss3;
          ss3 << std::string(title.size(), ' ')
              << warmDeltaT + sampleDeltaT
              << " seconds (Total)";
          writer(ss3.str());

          writer();
        }

        /**
         * Print timing information to all streams
         *
         * @param[in] warmDeltaT warmup time (sec)
         * @param[in] sampleDeltaT sample time (sec)
         */
        void write_timing(double warmDeltaT, double sampleDeltaT) {
          write_timing(warmDeltaT, sampleDeltaT, sample_writer_);
          write_timing(warmDeltaT, sampleDeltaT, diagnostic_writer_);
          write_timing(warmDeltaT, sampleDeltaT, message_writer_);
        }
      };

    }
  }
}
#endif
