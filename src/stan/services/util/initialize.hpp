#ifndef STAN_SERVICES_UTIL_INITIALIZE_HPP
#define STAN_SERVICES_UTIL_INITIALIZE_HPP

#include <stan/callbacks/writer.hpp>
#include <stan/io/var_context.hpp>
#include <stan/io/random_var_context.hpp>
#include <stan/io/chained_var_context.hpp>
#include <stan/model/log_prob_grad.hpp>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace stan {
  namespace services {
    namespace util {

      template <class Model, class RNG>
      std::vector<double> initialize(Model& model,
                                     stan::io::var_context& init,
                                     RNG& rng,
                                     double init_radius,
                                     bool print_timing,
                                     callbacks::writer&
                                     message_writer,
                                     callbacks::writer&
                                     init_writer) {
        std::vector<double> unconstrained;
        std::vector<int> disc_vector;
        std::stringstream msg;

        bool is_fully_initialized = true;
        bool any_initialized = false;
        std::vector<std::string> param_names;
        model.get_param_names(param_names);
        for (size_t n = 0; n < param_names.size(); n++) {
          is_fully_initialized &= init.contains_r(param_names[n]);
          any_initialized |= init.contains_r(param_names[n]);
        }

        bool init_zero = init_radius <= std::numeric_limits<double>::min();

        int MAX_INIT_TRIES = is_fully_initialized || init_zero ? 1 : 100;
        int num_init_tries = 0;
        for (; num_init_tries < MAX_INIT_TRIES; num_init_tries++) {
          stan::io::random_var_context
            random_context(model, rng, init_radius, init_zero);

          if (!any_initialized) {
            unconstrained = random_context.get_unconstrained();
          } else {
            stan::io::chained_var_context context(init, random_context);

            model.transform_inits(context,
                                  disc_vector,
                                  unconstrained,
                                  &msg);
            if (msg.str().length() > 0)
              message_writer(msg.str());
          }
          double log_prob(0);
          try {
            msg.str("");
            log_prob = model.template log_prob<false, true>
              (unconstrained, disc_vector, &msg);
            if (msg.str().length() > 0)
              message_writer(msg.str());
          } catch (std::exception& e) {
            message_writer();
            message_writer("Rejecting initial value:");
            message_writer("  Error evaluating the log probability "
                   "at the initial value.");
            continue;
          }
          if (!boost::math::isfinite(log_prob)) {
            message_writer("Rejecting initial value:");
            message_writer("  Log probability evaluates to log(0), "
                   "i.e. negative infinity.");
            message_writer("  Stan can't start sampling from this "
                           "initial value.");
            continue;
          }
          msg.str("");
          std::vector<double> gradient;
          bool gradient_ok = true;
          clock_t start_check = clock();
          log_prob = stan::model::log_prob_grad<true, true>
            (model, unconstrained, disc_vector,
             gradient, &msg);
          clock_t end_check = clock();
          double deltaT = static_cast<double>(end_check - start_check)
            / CLOCKS_PER_SEC;
          if (msg.str().length() > 0)
            message_writer(msg.str());

          for (size_t i = 0; i < gradient.size(); ++i) {
            if (gradient_ok && !boost::math::isfinite(gradient[i])) {
              message_writer("Rejecting initial value:");
              message_writer("  Gradient evaluated at the initial value "
                     "is not finite.");
              message_writer("  Stan can't start sampling from this "
                             "initial value.");
              gradient_ok = false;
            }
          }
          if (gradient_ok && print_timing) {
            msg.str("");
            message_writer();
            msg << "Gradient evaluation took " << deltaT << " seconds";
            message_writer(msg.str());
            msg.str("");
            msg  << "1000 transitions using 10 leapfrog steps "
                 << "per transition would take "
                 << 1e4 * deltaT << " seconds.";
            message_writer(msg.str());
            msg.str("");
            message_writer("Adjust your expectations accordingly!");
            message_writer();
            message_writer();
          }
          if (gradient_ok)
            break;
        }

        if (num_init_tries == MAX_INIT_TRIES) {
          if (!is_fully_initialized && !init_zero) {
            message_writer();
            msg.str("");
            msg << "Initialization between (-" << init_radius
                << ", " << init_radius << ") failed after "
                << MAX_INIT_TRIES <<  " attempts. ";
            message_writer(msg.str());
            message_writer(" Try specifying initial values,"
                           " reducing ranges of constrained values,"
                           " or reparameterizing the model.");
          }
          throw std::domain_error("");
        }

        init_writer(unconstrained);
        return unconstrained;
      }

    }
  }
}

#endif
