#ifndef STAN_SERVICES_UTIL_RUN_ADAPTIVE_SAMPLER_HPP
#define STAN_SERVICES_UTIL_RUN_ADAPTIVE_SAMPLER_HPP

#include <stan/services/util/generate_transitions.hpp>
#include <stan/services/util/mcmc_writer.hpp>
#include <ctime>
#include <vector>

namespace stan {
  namespace services {
    namespace util {

      template <class Sampler, class Model, class RNG>
      void run_adaptive_sampler(Sampler& sampler,
                                Model& model,
                                std::vector<double>& cont_vector,
                                int num_warmup,
                                int num_samples,
                                int num_thin,
                                int refresh,
                                bool save_warmup,
                                RNG& rng,
                                interface_callbacks::interrupt::base_interrupt&
                                interrupt,
                                interface_callbacks::writer::base_writer&
                                message_writer,
                                interface_callbacks::writer::base_writer&
                                error_writer,
                                interface_callbacks::writer::base_writer&
                                sample_writer,
                                interface_callbacks::writer::base_writer&
                                diagnostic_writer) {
        Eigen::Map<Eigen::VectorXd> cont_params(cont_vector.data(),
                                                cont_vector.size());

        sampler.engage_adaptation();
        try {
          sampler.z().q = cont_params;
          sampler.init_stepsize(message_writer, error_writer);
        } catch (const std::exception& e) {
          message_writer("Exception initializing step size.");
          message_writer(e.what());
          return;
        }

        services::sample::mcmc_writer
          writer(sample_writer, diagnostic_writer, message_writer);
        stan::mcmc::sample s(cont_params, 0, 0);

        // Headers
        writer.write_sample_names(s, sampler, model);
        writer.write_diagnostic_names(s, sampler, model);

        clock_t start = clock();
        stan::services::util::generate_transitions
          (sampler, num_warmup, 0, num_warmup + num_samples, num_thin,
           refresh, save_warmup, true,
           writer,
           s, model, rng,
           interrupt, message_writer, error_writer);
        clock_t end = clock();
        double warm_delta_t = static_cast<double>(end - start) / CLOCKS_PER_SEC;

        sampler.disengage_adaptation();
        writer.write_adapt_finish(sampler);

        start = clock();
        stan::services::util::generate_transitions
          (sampler, num_samples, num_warmup, num_warmup + num_samples, num_thin,
           refresh, true, false,
           writer,
           s, model, rng,
           interrupt, message_writer, error_writer);
        end = clock();
        double sample_delta_t
          = static_cast<double>(end - start) / CLOCKS_PER_SEC;

        writer.write_timing(warm_delta_t, sample_delta_t);
      }
    }
  }
}
#endif
