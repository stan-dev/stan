#ifndef STAN_SERVICES_SAMPLE_RUN_SAMPLER_HPP
#define STAN_SERVICES_SAMPLE_RUN_SAMPLER_HPP

#include <stan/services/sample/generate_transitions.hpp>
#include <stan/services/sample/mcmc_writer.hpp>
#include <ctime>

namespace stan {
  namespace services {
    namespace sample {
      template <class Model, class rng_t>
      void run_sampler(stan::mcmc::base_mcmc& sampler, Model& model,
                       Eigen::VectorXd& cont_params,
                       int num_warmup,
                       int num_samples,
                       int num_thin,
                       int refresh,
                       bool save_warmup,
                       bool adapt_engaged,
                       rng_t& base_rng,
                       interface_callbacks::interrupt::base_interrupt& interrupt,
                       interface_callbacks::writer::base_writer& sample_writer,
                       interface_callbacks::writer::base_writer& diagnostic_writer,
                       interface_callbacks::writer::base_writer& message_writer) {
        stan::services::sample::mcmc_writer<Model,
                                            interface_callbacks::writer::base_writer,
                                            interface_callbacks::writer::base_writer,
                                            interface_callbacks::writer::base_writer>
          writer(sample_writer, diagnostic_writer, message_writer);
        stan::mcmc::sample s(cont_params, 0, 0);

        // Headers
        writer.write_sample_names(s, sampler, model);
        writer.write_diagnostic_names(s, sampler, model);

        clock_t start = clock();
        generate_transitions(sampler, 
                             num_warmup, 0, num_warmup + num_samples,
                             num_thin, refresh, save_warmup, true,
                             writer, s,
                             model, base_rng, interrupt, message_writer);
        clock_t end = clock();
        double warm_delta_t = static_cast<double>(end - start) / CLOCKS_PER_SEC;

        if (adapt_engaged) {
          sampler.disengage_adaptation();
          writer.write_adapt_finish(sampler);
        }

        start = clock();
        generate_transitions(sampler, 
                             num_samples, num_warmup, num_warmup + num_samples,
                             num_thin, refresh, true, true,
                             writer, s,
                             model, base_rng, interrupt, message_writer);
        end = clock();
        double sample_delta_t = static_cast<double>(end - start) / CLOCKS_PER_SEC;

        writer.write_timing(warm_delta_t, sample_delta_t);
      }
    }
  }
}

#endif
