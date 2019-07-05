#ifndef STAN_MODEL_MODEL_CRTP_HPP
#define STAN_MODEL_MODEL_CRTP_HPP

#include <stan/io/var_context.hpp>
#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/model/model_base.hpp>
#include <iostream>
#include <utility>
#include <vector>

namespace stan {
namespace model {

// generated class foo will be declared as struct foo : public model_crtp<foo>
// type M requires:
//   - templated log_prob function
//   - templated write_array
// to be usable, require all virtuals in model_base


/**
 * Superclass for generated model class to adapt templated `log_prob`
 * and `write_array` methods to implement untemplated virtual
 * counterparts in the superclass `model_base`.
 *
 * <p>This class uses the curiously recursive template pattern (CRTP)
 * in order to adapt the templated methods.  As such, the derived
 * class `M` should be declared to extend `model_crtp<M>`; for
 * example, a model class `foo` should be declared as `foo : public
 * model_crtp<foo>`.
 *
 * @tparam M type of derived model, which must implemented concept
 * described in the class documentation
 */
template <typename M>
class model_crtp : public model_base {
 public:
  explicit model_crtp(size_t num_params_r) :
      model_base(num_params_r) { }

  virtual ~model_crtp() { }

  inline double log_prob(Eigen::VectorXd& theta,
                         std::ostream* msgs) const {
    return static_cast<const M*>(this)
        ->template log_prob<false, false, double>(theta, msgs);
  }
  inline math::var log_prob(Eigen::Matrix<math::var, -1, 1>& theta,
                            std::ostream* msgs) const {
    return static_cast<const M*>(this)
        ->template log_prob<false, false>(theta, msgs);
  }

  inline double log_prob_jacobian(Eigen::VectorXd& theta,
                                  std::ostream* msgs) const {
    return static_cast<const M*>(this)
        ->template log_prob<false, true>(theta, msgs);
  }
  inline math::var log_prob_jacobian(Eigen::Matrix<math::var, -1, 1>& theta,
                                  std::ostream* msgs) const {
    return static_cast<const M*>(this)
        ->template log_prob<false, true>(theta, msgs);
  }

  inline double log_prob_propto(Eigen::VectorXd& theta,
                                std::ostream* msgs) const {
    return static_cast<const M*>(this)
        ->template log_prob<true, false>(theta, msgs);
  }
  inline math::var log_prob_propto(Eigen::Matrix<math::var, -1, 1>& theta,
                                   std::ostream* msgs) const {
    return static_cast<const M*>(this)
        ->template log_prob<true, false>(theta, msgs);
  }

  inline double log_prob_propto_jacobian(Eigen::VectorXd& theta,
                                         std::ostream* msgs) const {
    return static_cast<const M*>(this)
        ->template log_prob<true, true>(theta, msgs);
  }
  inline math::var
  log_prob_propto_jacobian(Eigen::Matrix<math::var, -1, 1>& theta,
                           std::ostream* msgs) const {
    return static_cast<const M*>(this)
        ->template log_prob<true, true>(theta, msgs);
  }

  void write_array(boost::ecuyer1988& rng,
                   Eigen::VectorXd& theta,
                   Eigen::VectorXd& vars,
                   bool include_tparams = true,
                   bool include_gqs = true,
                   std::ostream* msgs = 0) const {
    return static_cast<const M*>(this)->template write_array(rng, theta, vars,
                                              include_tparams, include_gqs,
                                              msgs);
  }

  // TODO(carpenter): remove redundant std::vector methods below here =====
  // ======================================================================

  inline double log_prob(std::vector<double>& theta,
                         std::vector<int>& theta_i,
                         std::ostream* msgs) const {
    return static_cast<const M*>(this)
        ->template log_prob<false, false>(theta, theta_i, msgs);
  }
  inline math::var log_prob(std::vector<math::var>& theta,
                            std::vector<int>& theta_i,
                            std::ostream* msgs) const {
    return static_cast<const M*>(this)
        ->template log_prob<false, false>(theta, theta_i, msgs);
  }

  inline double log_prob_jacobian(std::vector<double>& theta,
                                  std::vector<int>& theta_i,
                                  std::ostream* msgs) const {
    return static_cast<const M*>(this)
        ->template log_prob<false, true>(theta, theta_i, msgs);
  }
  inline math::var log_prob_jacobian(std::vector<math::var>& theta,
                                     std::vector<int>& theta_i,
                                     std::ostream* msgs) const {
    return static_cast<const M*>(this)
        ->template log_prob<false, true>(theta, theta_i, msgs);
  }

  inline double log_prob_propto(std::vector<double>& theta,
                                std::vector<int>& theta_i,
                                std::ostream* msgs) const {
    return static_cast<const M*>(this)
        ->template log_prob<true, false>(theta, theta_i, msgs);
  }
  inline math::var log_prob_propto(std::vector<math::var>& theta,
                                   std::vector<int>& theta_i,
                                   std::ostream* msgs) const {
    return static_cast<const M*>(this)
        ->template log_prob<true, false>(theta, theta_i, msgs);
  }

  inline double log_prob_propto_jacobian(std::vector<double>& theta,
                                         std::vector<int>& theta_i,
                                         std::ostream* msgs) const {
    return static_cast<const M*>(this)
        ->template log_prob<true, true>(theta, theta_i, msgs);
  }
  inline math::var log_prob_propto_jacobian(std::vector<math::var>& theta,
                                            std::vector<int>& theta_i,
                                            std::ostream* msgs) const {
    return static_cast<const M*>(this)
        ->template log_prob<true, true>(theta, theta_i, msgs);
  }

  void write_array(boost::ecuyer1988& rng,
                   std::vector<double>& theta,
                   std::vector<int>& theta_i,
                   std::vector<double>& vars,
                   bool include_tparams = true,
                   bool include_gqs = true,
                   std::ostream* msgs = 0) const {
    return static_cast<const M*>(this)
        ->template write_array(rng, theta, theta_i, vars, include_tparams,
                               include_gqs, msgs);
  }
};

}  // namespace model
}  // namespace stan
#endif
