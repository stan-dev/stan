#ifndef STAN_SERVICES_SAMPLE_INIT_WINDOWED_ADAPT_HPP
#define STAN_SERVICES_SAMPLE_INIT_WINDOWED_ADAPT_HPP

#include <stan/services/arguments/categorical_argument.hpp>
#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/services/sample/init_adapt.hpp>

namespace stan {
  namespace services {
    namespace sample {
      /**
       * @tparam Sampler MCMC sampler implementation
       * @tparam ErrWriter An implementation of
       *                   src/stan/interface_callbacks/writer/base_writer.hpp
       * @param sampler MCMC sampler
       * @param adapt Adaptation configuration
       * @param num_warmup Number of warmup iterations
       * @param cont_params Continuous state values
       * @param err Writer callback for displaying error messages
       */
      template<class Sampler, class ErrWriter>
      bool init_windowed_adapt(Sampler& sampler,
                               stan::services::categorical_argument* adapt,
                               unsigned int num_warmup,
                               const Eigen::VectorXd& cont_params,
                               ErrWriter& err) {
        init_adapt<Sampler>(sampler, adapt, cont_params, err);

        unsigned int init_buffer
          = dynamic_cast<stan::services::u_int_argument*>
            (adapt->arg("init_buffer"))->value();
        unsigned int term_buffer
          = dynamic_cast<stan::services::u_int_argument*>
            (adapt->arg("term_buffer"))->value();
        unsigned int window
          = dynamic_cast<stan::services::u_int_argument*>
            (adapt->arg("window"))->value();

        sampler.set_window_params(num_warmup, init_buffer,
                                  term_buffer, window, err);

        return true;
      }

    }  // sample
  }  // services
}  // stan

#endif
