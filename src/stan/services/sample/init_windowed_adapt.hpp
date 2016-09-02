#ifndef STAN_SERVICES_SAMPLE_INIT_WINDOWED_ADAPT_HPP
#define STAN_SERVICES_SAMPLE_INIT_WINDOWED_ADAPT_HPP

#include <stan/interface_callbacks/writer/base_writer.hpp>
#include <stan/mcmc/base_mcmc.hpp>
#include <stan/services/arguments/categorical_argument.hpp>
#include <stan/services/sample/init_adapt.hpp>
#include <Eigen/Dense>
#include <ostream>

namespace stan {
  namespace services {
    namespace sample {

      template<class Sampler>
      bool
      init_windowed_adapt(stan::mcmc::base_mcmc* sampler,
                          stan::services::categorical_argument* adapt,
                          unsigned int num_warmup,
                          const Eigen::VectorXd& cont_params,
                          interface_callbacks::writer::base_writer& info_writer,
                      interface_callbacks::writer::base_writer& error_writer) {
        init_adapt<Sampler>(sampler, adapt, cont_params,
                            info_writer, error_writer);

        unsigned int init_buffer
          = dynamic_cast<u_int_argument*>(adapt->arg("init_buffer"))->value();
        unsigned int term_buffer
          = dynamic_cast<u_int_argument*>(adapt->arg("term_buffer"))->value();
        unsigned int window
          = dynamic_cast<u_int_argument*>(adapt->arg("window"))->value();

        dynamic_cast<Sampler*>(sampler)
          ->set_window_params(num_warmup, init_buffer, term_buffer,
                              window, info_writer);

        return true;
      }

    }
  }
}
#endif
