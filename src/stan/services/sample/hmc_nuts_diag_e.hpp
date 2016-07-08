#ifndef STAN_SERVICES_SAMPLE_HMC_NUTS_DIAG_E_HPP
#define STAN_SERVICES_SAMPLE_HMC_NUTS_DIAG_E_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/interface_callbacks/interrupt/base_interrupt.hpp>
#include <stan/interface_callbacks/writer/base_writer.hpp>
#include <stan/mcmc/fixed_param_sampler.hpp>
#include <stan/services/error_codes.hpp>
#include <stan/mcmc/hmc/nuts/diag_e_nuts.hpp>
#include <stan/services/util/run_sampler.hpp>
#include <stan/services/util/rng.hpp>
#include <stan/services/util/initialize.hpp>
#include <vector>

namespace stan {
  namespace services {
    namespace sample {

      template <class Model>
      int hmc_nuts_diag_e(Model& model,
                          stan::io::var_context& init,
                          unsigned int random_seed,
                          unsigned int chain,
                          double init_radius,
                          int num_warmup,
                          int num_samples,
                          int num_thin,
                          bool save_warmup,
                          int refresh,
                          double stepsize,
                          double stepsize_jitter,
                          int max_depth,
                          interface_callbacks::interrupt::base_interrupt&
                          interrupt,
                          interface_callbacks::writer::base_writer&
                          message_writer,
                          interface_callbacks::writer::base_writer&
                          error_writer,
                          interface_callbacks::writer::base_writer&
                          init_writer,
                          interface_callbacks::writer::base_writer&
                          sample_writer,
                          interface_callbacks::writer::base_writer&
                          diagnostic_writer) {
        boost::ecuyer1988 rng = stan::services::util::rng(random_seed, chain);

        std::vector<int> disc_vector;
        std::vector<double> cont_vector
          = stan::services::util::initialize(model, init, rng, init_radius,
                                             true,
                                             message_writer, init_writer);

        stan::mcmc::diag_e_nuts<Model, boost::ecuyer1988> sampler(model, rng);
        sampler.set_nominal_stepsize(stepsize);
        sampler.set_stepsize_jitter(stepsize_jitter);
        sampler.set_max_depth(max_depth);

        stan::services::util::run_sampler(sampler, model,
                                          cont_vector,
                                          num_warmup, num_samples, num_thin,
                                          refresh, save_warmup, rng,
                                          interrupt,
                                          message_writer, error_writer,
                                          sample_writer, diagnostic_writer);

        return stan::services::error_codes::OK;
      }

    }
  }
}
#endif
