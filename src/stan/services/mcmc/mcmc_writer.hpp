#ifndef STAN__SERVICES__MCMC__MCMC__WRITER__HPP
#define STAN__SERVICES__MCMC__MCMC__WRITER__HPP

#include <stan/mcmc/sample.hpp>
#include <stan/model/prob_grad.hpp>
#include <string>
#include <sstream>
#include <vector>

namespace stan {
  namespace services {
    namespace mcmc {

      template <class Model, class RNG,
                class SampleWriter, class DiagnosticWriter,
                class InfoWriter>
      class mcmc_writer {
      private:
        Model& model_;
        RNG& rng_;
        SampleWriter& sample_writer_;
        DiagnosticWriter& diagnostic_writer_;
        InfoWriter& info_writer_;

      public:

        mcmc_writer(Model& model,
                    RNG& rng,
                    SampleWriter& sample_writer,
                    DiagnosticWriter& diagnostic_writer,
                    InfoWriter& info_writer)
          : model_(model),
            rng_(rng),
            sample_writer_(sample_writer),
            diagnostic_writer_(diagnostic_writer),
            info_writer_(info_writer) {
        }

        template <class Sampler>
        void write_names(stan::mcmc::sample& sample, Sampler& sampler) {
          
          // Writer sample names
          std::vector<std::string> names;

          sample.get_sample_param_names(names);
          sampler.get_sampler_param_names(names);
          model_.constrained_param_names(names, true, true);

          sample_writer_.write_state_names(names);
          
          // Write diagnostic names
          names.clear();
          
          sample.get_sample_param_names(names);
          sampler.get_sampler_param_names(names);
          
          std::vector<std::string> model_names;
          model_.unconstrained_param_names(model_names, false, false);
          
          sampler.get_sampler_diagnostic_names(model_names, names);
          
          diagnostic_writer_.write_state_names(names);
          
        }

        template <class Sampler>
        void write_state(stan::mcmc::sample& sample, Sampler& sampler) {
          
          // External sampler state
          std::vector<double> values;

          sample.get_sample_params(values);
          sampler.get_sampler_params(values);

          Eigen::VectorXd model_values;

          model_.write_array(rng_,
                            const_cast<Eigen::VectorXd&>(sample.cont_params()),
                            model_values, true, true, &std::cout); // FIXME: remove cout in the future

          for (int i = 0; i < model_values.size(); ++i)
            values.push_back(model_values(i));

          sample_writer_.write_state(values);
          
          // Internal sampler state
          values.clear();
          
          sample.get_sample_params(values);
          sampler.get_sampler_params(values);
          sampler.get_sampler_diagnostics(values);
          
          diagnostic_writer_.write_state(values);
          
        }

        template <class Sampler>
        void write_adapt_finish(Sampler& sampler) {
          write_adapt_finish_(sampler, sample_writer_, "# ");
          write_adapt_finish_(sampler, diagnostic_writer_, "# ");
        }

        void write_timing(double warm_delta_t, double sample_delta_t) {
          write_timing_(warm_delta_t, sample_delta_t, sample_writer_, "# ");
          write_timing_(warm_delta_t, sample_delta_t, diagnostic_writer_, "# ");
          write_timing_(warm_delta_t, sample_delta_t, info_writer_, "");
        }
        
        void write_message(std::string& message) {
          info_writer_.write_message(message);
        }
        
      private:
        template <class Sampler, class Writer>
        void write_adapt_finish_(Sampler& sampler,
                                 Writer& writer, std::string prefix) {
          std::stringstream stream;
          sampler.write_sampler_state(&stream);
          
          writer.write_message(prefix + "Adaptation terminated");
          writer.write_message(prefix + stream.str());
        }
        
        template <class Writer>
        void write_timing_(double warm_delta_t, double sample_delta_t,
                           Writer& writer, std::string prefix) {
          writer.write_message("");
          writer.write_message(prefix + "Elapsed Time (seconds):");
          writer.write_key_value(prefix + "Warmup", warm_delta_t);
          writer.write_key_value(prefix + "Sampling", sample_delta_t);
          writer.write_key_value(prefix + "Total", warm_delta_t + sample_delta_t);
          writer.write_message("");
        }

      };
    } // mcmc
  } // services
} // stan

#endif
