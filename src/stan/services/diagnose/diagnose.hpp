#ifndef STAN_SERVICES_DIAGNOSE_DIAGNOSE_HPP
#define STAN_SERVICES_DIAGNOSE_DIAGNOSE_HPP

#include <stan/interface_callbacks/writer/base_writer.hpp>
#include <stan/math.hpp>
#include <stan/model/util.hpp>
#include <stan/services/error_codes.hpp>
#include <vector>

namespace stan {
  namespace services {
    namespace diagnose {

      /**
       * Checks the gradients of the model computed using reverse mode
       * autodiff against finite differences.
       *
       * This will test the first order gradients using reverse mode
       * at the value specified in cont_params. This method only
       * outputs to the message_writer.
       *
       * @tparam Model A model implementation
       *
       * @param model Input model to test (with data already instantiated)
       * @param cont_params Input values
       * @param epsilon epsilon to use for finite differences
       * @param error amount of absolute error to allow
       * @param message_writer Writer callback for display output
       * @param parameter_writer Writer callback for file output
       *
       * @return the number of parameters that are not within epsilon
       * of the finite difference calculation
       */
      template <class Model>
      int diagnose(Model& model,
                   Eigen::VectorXd& cont_params,
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
                                                  epsilon, error,
                                                  message_writer);

        return num_failed;
      }

    }
  }
}
#endif
