#ifndef STAN_SERVICES_SAMPLE_FIXED_PARAM_HPP
#define STAN_SERVICES_SAMPLE_FIXED_PARAM_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/interface_callbacks/interrupt/base_interrupt.hpp>
#include <stan/mcmc/fixed_param_sampler.hpp>
#include <stan/old_services/error_codes.hpp>
#include <stan/services/sample/mcmc_writer.hpp>
#include <stan/services/util/generate_transitions.hpp>
#include <stan/services/util/rng.hpp>
#include <stan/services/util/initialize.hpp>
#include <vector>

namespace stan {
  namespace services {
    namespace sample {

      template <class Model>
      int fixed_param(Model& model,
                      stan::io::var_context& init,
                      unsigned int random_seed,
                      unsigned int chain,
                      double init_radius,
                      int num_samples,
                      int num_thin,
                      int refresh,
                      interface_callbacks::interrupt::base_interrupt& interrupt,
                      interface_callbacks::writer::base_writer& message_writer,
                      interface_callbacks::writer::base_writer& error_writer,
                      interface_callbacks::writer::base_writer& sample_writer,
                      interface_callbacks::writer::base_writer&
                      diagnostic_writer) {
        boost::ecuyer1988 rng = stan::services::util::rng(random_seed, chain);

        std::vector<int> disc_vector;
        std::vector<double> cont_vector
          = stan::services::util::initialize(model, init, rng, init_radius,
                                             message_writer);

        stan::mcmc::fixed_param_sampler sampler;
        stan::services::sample::mcmc_writer
          writer(sample_writer, diagnostic_writer, message_writer);
        Eigen::VectorXd cont_params(cont_vector.size());
        for (size_t i = 0; i < cont_vector.size(); i++)
          cont_params[i] = cont_vector[i];
        stan::mcmc::sample s(cont_params, 0, 0);

        // Headers
        writer.write_sample_names(s, sampler, model);
        writer.write_diagnostic_names(s, sampler, model);

        clock_t start = clock();

        stan::services::util::generate_transitions
          (sampler, num_samples, 0, num_samples, num_thin,
           refresh, true, false,
           writer,
           s, model, rng,
           interrupt, message_writer, error_writer);
        clock_t end = clock();

        double sampleDeltaT = static_cast<double>(end - start) / CLOCKS_PER_SEC;
        writer.write_timing(0.0, sampleDeltaT);

        return stan::services::error_codes::OK;
      }

    }
  }
}
#endif
