#ifndef STAN_SERVICES_EXPERIMENTAL_ADVI_FULLRANK_HPP
#define STAN_SERVICES_EXPERIMENTAL_ADVI_FULLRANK_HPP

#include <stan/interface_callbacks/writer/base_writer.hpp>
#include <stan/services/util/experimental_message.hpp>
#include <stan/services/util/initialize.hpp>
#include <stan/services/util/rng.hpp>
#include <stan/io/var_context.hpp>
#include <stan/variational/advi.hpp>
#include <boost/random/additive_combine.hpp>
#include <string>
#include <vector>

namespace stan {
  namespace services {
    namespace experimental {
      namespace advi {

        template <class Model, typename Interrupt>
        int fullrank(Model& model,
                     stan::io::var_context& init,
                     unsigned int random_seed,
                     unsigned int chain,
                     double init_radius,
                     int grad_samples,
                     int elbo_samples,
                     int max_iterations,
                     double tol_rel_obj,
                     double eta,
                     bool adapt_engaged,
                     int adapt_iterations,
                     int eval_elbo,
                     int output_samples,
                     Interrupt& interrupt,
                     interface_callbacks::writer::base_writer& message_writer,
                     interface_callbacks::writer::base_writer& parameter_writer,
                     interface_callbacks::writer::base_writer&
                     diagnostic_writer) {
          stan::services::util::experimental_message(message_writer);

          boost::ecuyer1988 rng = stan::services::util::rng(random_seed, chain);

          std::vector<int> disc_vector;
          std::vector<double> cont_vector
            = stan::services::util::initialize(model, init, rng, init_radius,
                                               message_writer);


          std::vector<std::string> names;
          names.push_back("lp__");
          model.constrained_param_names(names, true, true);
          parameter_writer(names);

          Eigen::VectorXd cont_params;
          cont_params.resize(cont_vector.size());
          for (size_t n = 0; n < cont_vector.size(); n++)
            cont_params[n] = cont_vector[n];

          stan::variational::advi<Model,
                                  stan::variational::normal_fullrank,
                                  boost::ecuyer1988>
            cmd_advi(model,
                     cont_params,
                     rng,
                     grad_samples,
                     elbo_samples,
                     eval_elbo,
                     output_samples);
          cmd_advi.run(eta, adapt_engaged, adapt_iterations,
                       tol_rel_obj, max_iterations,
                       message_writer, parameter_writer, diagnostic_writer);

          return 0;
        }
      }
    }
  }
}

#endif
