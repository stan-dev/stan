#ifndef STAN_SERVICES_DIAGNOSE_DIAGNOSE_HPP
#define STAN_SERVICES_DIAGNOSE_DIAGNOSE_HPP

#include <stan/interface_callbacks/writer/base_writer.hpp>
#include <stan/math.hpp>
#include <stan/model/util.hpp>
#include <stan/services/error_codes.hpp>

namespace stan {
  namespace services {
    namespace diagnose {

      /**
       * @tparam Model A model implementation
       * @param cont_params Input values
       * @param model Input model
       * @param epsilon epsilon to use for finite differences
       * @param error amount of absolute error to allow
       * @param message_writer Writer callback for display output
       * @param parameter_writer Writer callback for file output
       */
      template <class Model>
      int diagnose(Eigen::VectorXd& cont_params,
                   Model& model,
                   double epsilon,
                   double error,
                   interface_callbacks::writer::base_writer& message_writer,
                   interface_callbacks::writer::base_writer& parameter_writer) {
        std::vector<double> cont_vector(cont_params.size());
        for (int i = 0; i < cont_params.size(); ++i)
          cont_vector.at(i) = cont_params(i);
        std::vector<int> disc_vector;

        message_writer();
        message_writer("TEST GRADIENT MODE");

        int num_failed =
          stan::model::test_gradients<true, true>(model,
                                                  cont_vector, disc_vector,
                                                  epsilon, error, message_writer);

        // FIXME: this is wasteful and runs the finite diff code twice
        num_failed =
          stan::model::test_gradients<true, true>(model,
                                                  cont_vector, disc_vector,
                                                  epsilon, error, parameter_writer);

        (void) num_failed;  // FIXME: do something with the number failed

        return stan::services::error_codes::OK;
      }

    }
  }
}
#endif
