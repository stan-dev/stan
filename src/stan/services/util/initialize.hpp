#ifndef STAN_SERVICES_UTIL_INITIALIZE_HPP
#define STAN_SERVICES_UTIL_INITIALIZE_HPP

#include <stan/io/var_context.hpp>
#include <stan/io/random_var_context.hpp>
#include <stan/io/chained_var_context.hpp>
#include <stan/model/util.hpp>
#include <sstream>
#include <vector>

namespace stan {
  namespace services {
    namespace util {


      template <class Model, class RNG>
      std::vector<double> initialize(Model& model,
                                     stan::io::var_context& init,
                                     RNG& rng,
                                     double init_radius,
                                     interface_callbacks::writer::base_writer&
                                     message_writer,
                                     interface_callbacks::writer::base_writer&
                                     init_writer) {
        std::vector<double> unconstrained;
        std::vector<int> disc_vector;
        std::stringstream msg;

        int MAX_INIT_TRIES = 100;
        int num_init_tries = 0;
        for (; num_init_tries < MAX_INIT_TRIES; num_init_tries++) {
          stan::io::random_var_context random_context(model, rng, init_radius);
          stan::io::chained_var_context context(init, random_context);
          double log_prob(0);

          model.transform_inits(context,
                                disc_vector,
                                unconstrained,
                                &msg);
          if (msg.str().length() > 0)
            message_writer(msg.str());
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
          }
          if (!boost::math::isfinite(log_prob)) {
            message_writer("Rejecting initial value:");
            message_writer("  Log probability evaluates to log(0), "
                   "i.e. negative infinity.");
            message_writer("  Stan can't start sampling from this "
                           "initial value.");
          }
          msg.str("");
          std::vector<double> gradient;
          bool gradient_ok = true;
          log_prob = stan::model::log_prob_grad<true, true>
            (model, unconstrained, disc_vector,
             gradient, &msg);
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
          if (gradient_ok)
            break;
        }

        if (num_init_tries == MAX_INIT_TRIES) {
          message_writer();
          msg.str("");
          msg << "Initialization between (-" << init_radius
              << ", " << init_radius << ") failed after "
              << MAX_INIT_TRIES <<  " attempts. ";
          message_writer(msg.str());
          message_writer(" Try specifying initial values,"
                         " reducing ranges of constrained values,"
                         " or reparameterizing the model.");

          throw std::domain_error("");
        }
        init_writer(unconstrained);
        return unconstrained;
      }

    }
  }
}

#endif
