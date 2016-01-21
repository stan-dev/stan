#ifndef STAN_SERVICES_SAMPLE_FIXED_PARAM_HPP
#define STAN_SERVICES_SAMPLE_FIXED_PARAM_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/interface_callbacks/interrupt/base_interrupt.hpp>
#include <stan/mcmc/fixed_param_sampler.hpp>
#include <stan/services/sample/mcmc_writer.hpp>
#include <stan/services/mcmc/sample.hpp>

namespace stan {
  namespace services {
    namespace sample {
      
      template <class Model, class rng_t>
      int fixed_param(Model& model,
                      rng_t& base_rng,
                      Eigen::VectorXd& cont_params,
                      int num_samples,
                      int num_thin,
                      int refresh,
                      interface_callbacks::interrupt::base_interrupt& interrupt,
                      interface_callbacks::writer::base_writer& sample_writer,
                      interface_callbacks::writer::base_writer& diagnostic_writer,
                      interface_callbacks::writer::base_writer& message_writer) {
        stan::mcmc::fixed_param_sampler sampler;
        stan::services::sample::mcmc_writer<Model,
                                            interface_callbacks::writer::base_writer,
                                            interface_callbacks::writer::base_writer,
                                            interface_callbacks::writer::base_writer>
          writer(sample_writer, diagnostic_writer, message_writer);
        stan::mcmc::sample s(cont_params, 0, 0);

        // Headers
        writer.write_sample_names(s, &sampler, model);
        writer.write_diagnostic_names(s, &sampler, model);
        
        clock_t start = clock();
        mcmc::sample<Model, rng_t>(&sampler, 0, num_samples, num_thin,
                                   refresh, true,
                                   writer,
                                   s, model, base_rng,
                                   interrupt,
                                   message_writer);        
        clock_t end = clock();

        double sampleDeltaT = static_cast<double>(end - start) / CLOCKS_PER_SEC;
        writer.write_timing(0.0, sampleDeltaT);
        
        return 0;
      }
      
    }
  }
}
#endif
