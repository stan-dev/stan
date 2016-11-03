#ifndef STAN_MODEL_GRADIENT_HPP
#define STAN_MODEL_GRADIENT_HPP

#include <stan/interface_callbacks/writer/base_writer.hpp>
#include <stan/math/rev/mat.hpp>
#include <stan/model/model_functional.hpp>
#include <sstream>
#include <stdexcept>

namespace stan {
  namespace model {

    template <class M>
    void gradient(const M& model,
                  const Eigen::Matrix<double, Eigen::Dynamic, 1>& x,
                  double& f,
                  Eigen::Matrix<double, Eigen::Dynamic, 1>& grad_f,
                  std::ostream* msgs = 0) {
      stan::math::gradient(model_functional<M>(model, msgs), x, f, grad_f);
    }

    template <class M>
    void gradient(const M& model,
                  const Eigen::Matrix<double, Eigen::Dynamic, 1>& x,
                  double& f,
                  Eigen::Matrix<double, Eigen::Dynamic, 1>& grad_f,
                  stan::interface_callbacks::writer::base_writer& writer) {
      std::stringstream ss;
      try {
        stan::math::gradient(model_functional<M>(model, &ss), x, f, grad_f);
      } catch (std::exception& e) {
        if (ss.str().length() > 0)
          writer(ss.str());
        throw;
      }
      if (ss.str().length() > 0)
        writer(ss.str());
    }

  }
}
#endif
