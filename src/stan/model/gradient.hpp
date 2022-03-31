#ifndef STAN_MODEL_GRADIENT_HPP
#define STAN_MODEL_GRADIENT_HPP

#include <stan/callbacks/logger.hpp>
#include <stan/callbacks/writer.hpp>
#include <stan/math/rev.hpp>
#include <stan/model/model_functional.hpp>
#include <sstream>
#include <stdexcept>

namespace stan {
namespace model {

template <class M>
inline void gradient(const M& model, const Eigen::Matrix<double, Eigen::Dynamic, 1>& x,
              double& f, Eigen::Matrix<double, Eigen::Dynamic, 1>& grad_f,
              std::ostream* msgs = 0) {
  try{
    {
    using stan::math::var;
    stan::math::nested_rev_autodiff nested;
    stan::math::var_value<Eigen::VectorXd> x_var(x);
    var fx_var = model.template log_prob<true, true>(x_var, msgs);
    f = fx_var.val();
    grad_f.resize(x.size());
    grad(fx_var.vi_);
    grad_f = x_var.adj();
    }
    stan::math::recover_memory();
  } catch (const std::exception& ex) {
    stan::math::recover_memory();
    throw ex;
  }
}

template <typename M>
inline void gradient(const M& model, const Eigen::Matrix<double, Eigen::Dynamic, 1>& x,
              double& f, Eigen::Matrix<double, Eigen::Dynamic, 1>& grad_f,
              callbacks::logger& logger) {
  std::stringstream ss;
  try {
    gradient(model, x, f, grad_f, &ss);
  } catch (std::exception& e) {
    if (ss.str().length() > 0)
      logger.info(ss);
    throw;
  }
  if (ss.str().length() > 0)
    logger.info(ss);
}

}  // namespace model
}  // namespace stan
#endif
