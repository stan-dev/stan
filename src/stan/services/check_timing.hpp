#ifndef STAN_SERVICES_CHECK_TIMING_HPP
#define STAN_SERVICES_CHECK_TIMING_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/interface_callbacks/writer/base_writer.hpp>
#include <stan/model/util.hpp>
#include <ctime>
#include <sstream>

namespace stan {
  namespace services {

    /**
     * Checks timing of the model by calculating the gradient once.
     *
     * @tparam Model class of the model
     * @param model Instance of the model
     * @param cont_params Initial parameter values
     * @param message_writer Writer to output the timing results
     */
    template <class Model>
    void check_timing(Model& model,
                      const Eigen::VectorXd& cont_params,
                      interface_callbacks::writer::base_writer&
                      message_writer) {
      double init_log_prob = 0;
      Eigen::VectorXd init_grad = Eigen::VectorXd::Zero(model.num_params_r());

      clock_t start = clock();

      stan::model::gradient(model, cont_params, init_log_prob,
                            init_grad);

      clock_t end = clock();
      double deltaT = static_cast<double>(end - start) / CLOCKS_PER_SEC;

      std::stringstream msg;
      message_writer();
      msg << "Gradient evaluation took " << deltaT << " seconds";
      message_writer(msg.str());
      msg.str("");
      msg << "1000 transitions using 10 leapfrog steps "
          << "per transition would take "
          << 1e4 * deltaT << " seconds.";
      message_writer(msg.str());
      message_writer("Adjust your expectations accordingly!");
      message_writer();
      message_writer();

      return;
    }
  }

}

#endif
