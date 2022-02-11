#ifndef STAN_MODEL_MODEL_BASE_CRTP_HPP
#define STAN_MODEL_MODEL_BASE_CRTP_HPP

#include <stan/model/model_base.hpp>
#include <iostream>
#include <utility>
#include <vector>

namespace stan {
namespace model {

/**
 * Base class employing the curiously recursive template pattern for
 * static inheritance to adapt templated `log_prob` and `write_array`
 * methods to their untemplated virtual counterparts declared in
 * `model_base`.
 *
 * The derived class `M` is required to implement the following two
 * pairs of template functions,
 *
 * ```
 * template <bool propto, bool jacobian, typename T>
 * T log_prob(std::vector<T>& params_r,
 *            std::vector<int>& params_i,
 *            std::ostream* msgs = 0) const;

 * template <bool propto, bool jacobian, typename T>
 * T log_prob(Eigen::Matrix<T, -1, 1>& params_r,
 *            std::ostream* msgs = 0) const;
 * ```
 *
 * and
 *
 * ```
 * template <typename RNG>
 * void write_array(RNG& base_rng,
 *                  std::vector<double>& params_r,
 *                  std::vector<int>& params_i,
 *                  std::vector<double>& vars,
 *                  bool include_tparams = true,
 *                  bool include_gqs = true,
 *                  std::ostream* msgs = 0) const;
 *
 * template <typename RNG>
 * void write_array(RNG& base_rng,
 *                  Eigen::Matrix<double, -1, 1>& params_r,
 *                  Eigen::Matrix<double, -1, 1>& vars,
 *                  bool include_tparams = true,
 *                  bool include_gqs = true,
 *                  std::ostream* msgs = 0) const
 * ```
 *
 * <p>The derived class `M` must be declared following the curiously
 * recursive template pattern, for example, if `M` is `foo_model`,
 * then `foo_model` should be declared as
 *
 * ```
 * class foo_model : public stan::model::model_base_crtp<foo_model> { ... };
 * ```
 *
 * The recursion arises when the type of the declared class appears as
 * a template parameter in the class it extends.  For example,
 * `foo_model` is declared to extend `model_base_crtp<foo_model>`.  In
 * general, the template parameter `M` for this class is called the
 * derived class, and must be declared to extend `foo_model<M>`.
 *
 * @tparam M type of derived model, which must implemented the
 * template methods defined in the class documentation
 */
template <typename M>
class model_base_crtp : public stan::model::model_base {
 public:
  /**
   * Construct a model with the specified number of real unconstrained
   * parameters.
   *
   * @param[in] num_params_r number of real unconstrained parameters
   */
  explicit model_base_crtp(size_t num_params_r) : model_base(num_params_r) {}

  /**
   * Destroy this class.  This is required to be virtual to allow
   * subclass references to clean up superclasses, but is otherwise a
   * no-op.
   */
  virtual ~model_base_crtp() {}

  inline double log_prob(Eigen::VectorXd& theta,
                         std::ostream* msgs) const override {
    return static_cast<const M*>(this)->template log_prob<false, false, double>(
        theta, msgs);
  }
  inline math::var log_prob(Eigen::Matrix<math::var, -1, 1>& theta,
                            std::ostream* msgs) const override {
    return static_cast<const M*>(this)->template log_prob<false, false>(theta,
                                                                        msgs);
  }

  inline double log_prob_jacobian(Eigen::VectorXd& theta,
                                  std::ostream* msgs) const override {
    return static_cast<const M*>(this)->template log_prob<false, true>(theta,
                                                                       msgs);
  }
  inline math::var log_prob_jacobian(Eigen::Matrix<math::var, -1, 1>& theta,
                                     std::ostream* msgs) const override {
    return static_cast<const M*>(this)->template log_prob<false, true>(theta,
                                                                       msgs);
  }

  inline double log_prob_propto(Eigen::VectorXd& theta,
                                std::ostream* msgs) const override {
    return static_cast<const M*>(this)->template log_prob<true, false>(theta,
                                                                       msgs);
  }
  inline math::var log_prob_propto(Eigen::Matrix<math::var, -1, 1>& theta,
                                   std::ostream* msgs) const override {
    return static_cast<const M*>(this)->template log_prob<true, false>(theta,
                                                                       msgs);
  }

  inline double log_prob_propto_jacobian(Eigen::VectorXd& theta,
                                         std::ostream* msgs) const override {
    return static_cast<const M*>(this)->template log_prob<true, true>(theta,
                                                                      msgs);
  }
  inline math::var log_prob_propto_jacobian(
      Eigen::Matrix<math::var, -1, 1>& theta,
      std::ostream* msgs) const override {
    return static_cast<const M*>(this)->template log_prob<true, true>(theta,
                                                                      msgs);
  }

  void write_array(boost::ecuyer1988& rng, Eigen::VectorXd& theta,
                   Eigen::VectorXd& vars, bool include_tparams = true,
                   bool include_gqs = true,
                   std::ostream* msgs = 0) const override {
    return static_cast<const M*>(this)->template write_array(
        rng, theta, vars, include_tparams, include_gqs, msgs);
  }

  // TODO(carpenter): remove redundant std::vector methods below here =====
  // ======================================================================

  inline double log_prob(std::vector<double>& theta, std::vector<int>& theta_i,
                         std::ostream* msgs) const override {
    return static_cast<const M*>(this)->template log_prob<false, false>(
        theta, theta_i, msgs);
  }
  inline math::var log_prob(std::vector<math::var>& theta,
                            std::vector<int>& theta_i,
                            std::ostream* msgs) const override {
    return static_cast<const M*>(this)->template log_prob<false, false>(
        theta, theta_i, msgs);
  }

  inline double log_prob_jacobian(std::vector<double>& theta,
                                  std::vector<int>& theta_i,
                                  std::ostream* msgs) const override {
    return static_cast<const M*>(this)->template log_prob<false, true>(
        theta, theta_i, msgs);
  }
  inline math::var log_prob_jacobian(std::vector<math::var>& theta,
                                     std::vector<int>& theta_i,
                                     std::ostream* msgs) const override {
    return static_cast<const M*>(this)->template log_prob<false, true>(
        theta, theta_i, msgs);
  }

  inline double log_prob_propto(std::vector<double>& theta,
                                std::vector<int>& theta_i,
                                std::ostream* msgs) const override {
    return static_cast<const M*>(this)->template log_prob<true, false>(
        theta, theta_i, msgs);
  }
  inline math::var log_prob_propto(std::vector<math::var>& theta,
                                   std::vector<int>& theta_i,
                                   std::ostream* msgs) const override {
    return static_cast<const M*>(this)->template log_prob<true, false>(
        theta, theta_i, msgs);
  }

  inline double log_prob_propto_jacobian(std::vector<double>& theta,
                                         std::vector<int>& theta_i,
                                         std::ostream* msgs) const override {
    return static_cast<const M*>(this)->template log_prob<true, true>(
        theta, theta_i, msgs);
  }
  inline math::var log_prob_propto_jacobian(std::vector<math::var>& theta,
                                            std::vector<int>& theta_i,
                                            std::ostream* msgs) const override {
    return static_cast<const M*>(this)->template log_prob<true, true>(
        theta, theta_i, msgs);
  }

  void write_array(boost::ecuyer1988& rng, std::vector<double>& theta,
                   std::vector<int>& theta_i, std::vector<double>& vars,
                   bool include_tparams = true, bool include_gqs = true,
                   std::ostream* msgs = 0) const override {
    return static_cast<const M*>(this)->template write_array(
        rng, theta, theta_i, vars, include_tparams, include_gqs, msgs);
  }

  void transform_inits(const io::var_context& context,
                       Eigen::VectorXd& params_r,
                       std::ostream* msgs) const override {
    return static_cast<const M*>(this)->transform_inits(context, params_r,
                                                        msgs);
  }
};

}  // namespace model
}  // namespace stan
#endif
