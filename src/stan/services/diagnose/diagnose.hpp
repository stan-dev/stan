#ifndef STAN_SERVICES_DIAGNOSE_DIAGNOSE_HPP
#define STAN_SERVICES_DIAGNOSE_DIAGNOSE_HPP

#include <stan/model/util.hpp>
#include <stan/services/arguments/list_argument.hpp>
#include <stan/services/arguments/singleton_argument.hpp>
#include <stan/services/error_codes.hpp>
#include <vector>

namespace stan {
  namespace services {
    namespace diagnose {
      /**
       * @tparam Model A model implementation
       * @tparam InfoWriter An implementation of
       *                    src/stan/interface_callbacks/writer/base_writer.hpp
       * @tparam OutputWriter An implementation of
       *                      src/stan/interface_callbacks/writer/base_writer.hpp
       * @param cont_params Input values
       * @param model Input model
       * @param test Diagnostic configuration
       * @param info Writer callback for display output
       * @param output Writer callback for file output
       */
      template <class Model, class InfoWriter, class OutputWriter>
      int diagnose(Eigen::VectorXd& cont_params,
                   Model& model,
                   stan::services::list_argument* test,
                   InfoWriter& info,
                   OutputWriter& output) {
        std::vector<double> cont_vector(cont_params.size());
        for (int i = 0; i < cont_params.size(); ++i)
          cont_vector.at(i) = cont_params(i);
        std::vector<int> disc_vector;

        if (test->value() == "gradient") {
          info();
          info("TEST GRADIENT MODE");

          double epsilon = dynamic_cast<stan::services::real_argument*>
                           (test->arg("gradient")->arg("epsilon"))->value();

          double error = dynamic_cast<stan::services::real_argument*>
                         (test->arg("gradient")->arg("error"))->value();

          int num_failed =
            stan::model::test_gradients<true, true>(model,
                                                    cont_vector, disc_vector,
                                                    info, epsilon, error);

          num_failed =
            stan::model::test_gradients<true, true>(model,
                                                    cont_vector, disc_vector,
                                                    output, epsilon, error);

          (void) num_failed;  // FIXME: do something with the number failed

          return stan::services::error_codes::OK;
        }

        return stan::services::error_codes::USAGE;
      }

    }  // namespace diagnose
  }  // namespace services
}  // namespace stan


#endif
