#ifndef STAN_SERVICES_OPTIMIZE_NEWTON_HPP
#define STAN_SERVICES_OPTIMIZE_NEWTON_HPP

#include <stan/interface_callbacks/writer/base_writer.hpp>
#include <stan/math.hpp>
#include <stan/optimization/newton.hpp>
#include <stan/services/error_codes.hpp>
#include <stan/services/io/write_iteration.hpp>
#include <stan/services/io/write_error_msg.hpp>
#include <cmath>

namespace stan {
  namespace services {
    namespace optimize {

      /**
       * Runs the Newton algorithm for a model.
       * 
       * @tparam Model Stan model class
       * @tparam RNG Random number generator class
       * @tparam Interrupt callback for interrupts
       * @param[in] model the Stan model
       * @param[in] base_rng random number generator
       * @param[in, out] cont_params continuous parameters; starts at the 
       *   initial value. Ends at the optimum. This must be the same size as
       *   the model parameter size.
       * @param[in] num_iterations maximum number of iterations
       * @param[in] save_iterations indicates whether all the interations should
       *   be saved
       * @param[out] message_writer output for messages
       * @param[out] parameter_writer output for parameter values
       * @return stan::services::error_codes::OK (0) if successful
       */
      template <class Model, class RNG, class Interrupt>
      int newton(Model& model, RNG& base_rng,
                 Eigen::VectorXd& cont_params,
                 int num_iterations,
                 bool save_iterations,
                 Interrupt& interrupt,
                 interface_callbacks::writer::base_writer& message_writer,
                 interface_callbacks::writer::base_writer& parameter_writer) {
        std::stringstream message;
        std::vector<double> cont_vector(cont_params.size());
        std::vector<int> disc_vector;
        for (int i = 0; i < cont_params.size(); ++i)
          cont_vector[i] = cont_params[i];
        double lp = 0;

        try {
          lp = model.template log_prob<false, false>(cont_vector, disc_vector,
                                                     &message);
          message_writer(message.str());
        } catch (const std::exception& e) {
          services::io::write_error_msg(message_writer, e);
          lp = -std::numeric_limits<double>::infinity();
        }

        message.str("");
        message << "Initial log joint probability = " << lp;
        message_writer(message.str());

        std::vector<std::string> names;
        names.push_back("lp__");
        model.constrained_param_names(names, true, true);
        parameter_writer(names);


        double lastlp = lp;
        for (int m = 0; m < num_iterations; m++) {
          if (save_iterations)
            io::write_iteration(model, base_rng,
                                lp, cont_vector, disc_vector,
                                message_writer, parameter_writer);
          interrupt();
          lastlp = lp;
          lp = stan::optimization::newton_step(model, cont_vector, disc_vector);

          message.str("");
          message << "Iteration "
                  << std::setw(2) << (m + 1) << ". "
                  << "Log joint probability = " << std::setw(10) << lp
                  << ". Improved by " << (lp - lastlp) << ".";
          message_writer(message.str());

          if (std::fabs(lp - lastlp) <= 1e-8)
            break;
        }

        io::write_iteration(model, base_rng,
                            lp, cont_vector, disc_vector,
                            message_writer, parameter_writer);
        for (int i = 0; i < cont_params.size(); ++i)
          cont_params[i] = cont_vector[i];
        return stan::services::error_codes::OK;
      }

    }
  }
}
#endif
