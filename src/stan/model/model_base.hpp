#ifndef STAN_MODEL_MODEL_BASE_HPP
#define STAN_MODEL_MODEL_BASE_HPP

#include <stan/io/var_context.hpp>
#include <stan/math/rev/core.hpp>
#ifdef STAN_MODEL_FVAR_VAR
#include <stan/math/mix.hpp>
#endif
#include <stan/model/prob_grad.hpp>
#include <boost/random/additive_combine.hpp>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

namespace stan {
namespace model {

/**
 * The base class for models defining all virtual methods required for
 * services.  Any class extending this class and defining all of its
 * virtual methods can be used with any of the Stan services for
 * sampling, optimization, or variational inference.
 *
 * <p><i>Implementation Details:</i> The reason there are so many
 * overloads of the `log_prob` and `write_array` methods is that
 * template methods cannot be declared virtual.  This class extends
 * `stan::model::prob_grad` in order to define sizing for the number
 * of unconstrained parameters;  thus it is not a pure virtual base
 * class.
 *
 *<p>The approach to defining models used by the Stan language code
 * generator is use the curiously recursive template base class defined
 * in the extension `stan::model::model_base_crtp`.
 */
class model_base : public prob_grad {
 public:
  /**
   * Construct a model with the specified number of real valued
   * unconstrained parameters.
   *
   * @param[in] num_params_r number of real-valued, unconstrained
   * parameters
   */
  explicit model_base(size_t num_params_r) : prob_grad(num_params_r) {}

  /**
   * Destructor.  This class has a no-op destructor.
   */
  virtual ~model_base() {}

  /**
   * Return the name of the model.
   *
   * @return model name
   */
  virtual std::string model_name() const = 0;

  /**
   * Returns the compile information of the model:
   * stanc version and stanc flags used to compile the model.
   *
   * @return model name
   */
  virtual std::vector<std::string> model_compile_info() const = 0;

  /**
   * Set the specified argument to sequence of parameters, transformed
   * parameters, and generated quantities in the order in which they
   * were declared.  The input sequence is cleared and resized.
   *
   * @param[in,out] names sequence of names parameters, transformed
   * parameters, and generated quantities
   */
  virtual void get_param_names(std::vector<std::string>& names) const = 0;

  /**
   * Set the dimensionalities of constrained parameters, transformed
   * parameters, and generated quantities.  The input sequence is
   * cleared and resized.  The dimensions of each parameter
   * dimensionality is represented by a sequence of sizes.  Scalar
   * real and integer parameters are represented as an empty sequence
   * of dimensions.
   *
   * <p>Indexes are output in the order they are used in indexing. For
   * example, a 2 x 3 x 4 array will have dimensionality
   * `std::vector<size_t>{ 2, 3, 4 }`, whereas a 2-dimensional array
   * of 3-vectors will have dimensionality `std::vector<size_t>{ 2, 3
   * }`, and a 2-dimensional array of 3 x 4 matrices will have
   * dimensionality `std::vector<size_t>{2, 3, 4}`.
   *
   * @param[in,out] dimss sequence of dimensions specifications to set
   */
  virtual void get_dims(std::vector<std::vector<size_t> >& dimss) const = 0;

  /**
   *  Set the specified sequence to the indexed, scalar, constrained
   *  parameter names.  Each variable is output with a
   *  period-separated list of indexes as a suffix, indexing from 1.
   *
   * <p>A real parameter `alpha` will produce output `alpha` with no
   * indexes.
   *
   * <p>A 3-dimensional vector (row vector, simplex, etc.)
   * `theta` will produce output `theta.1`, `theta.2`, `theta.3`.  The
   * dimensions are the constrained dimensions.
   *
   * <p>Matrices are output column major to match their internal
   * representation in the Stan math library, so that a 2 x 3 matrix
   * `X` will produce output `X.1.1, X.2.1, X.1.2, X.2.2, X.1.3,
   * X.2.3`.
   *
   * <p>Arrays are handled in natural C++ order, 2 x 3 x 4 array `a`
   * will produce output `a.1.1.1`, `a.1.1.2`, `a.1.1.3`, `a.1.1.4`,
   * `a.1.2.1`, `a.1.2.2`, `a.1.2.3`, `a.1.2.4`, `a.1.3.1`, `a.1.3.2`,
   * `a.1.3.3`, `a.1.3.4`, `a.2.1.1`, `a.2.1.2`, `a.2.1.3`, `a.2.1.4`,
   * `a.2.2.1`, `a.2.2.2`, `a.2.2.3`, `a.2.2.4`, `a.2.3.1`, `a.2.3.2`,
   * `a.2.3.3`, `a.2.3.4`.
   *
   * <p>Arrays of vectors are handled as expected, so that a
   * 2-dimensional array of 3-vectors is output as `B.1.1`, `B.2.1`,
   * `B.1.2`, `B.2.2`, `B.1.3`, `B.2.3`.

   * <p>Arrays of matrices are generated in row-major order for the
   * array components and column-major order for the matrix component.
   * Thus a 2-dimensional array of 3 by 4 matrices `B` will be of
   * dimensionality 2 x 3 x 4 (indexes `B[1:2, 1:3, 1:4]`) and will be output
   * as `B.1.1.1`, `B.2.1.1`, `B.1.2.1`, `B.2.2.1`, `B.1.3.1`,
   * `B.2.3.1`, `B.1.1.2`, `B.2.1.2`, `B.1.2.2`, `B.2.2.2`, `B.1.3.2`,
   * `B.2.3.2`, `B.1.1.3`, `B.2.1.3`, `B.1.2.3`, `B.2.2.3`, `B.1.3.3`,
   * `B.2.3.3`, `B.1.1.4`, `B.2.1.4`, `B.1.2.4`, `B.2.2.4`, `B.1.3.4`,
   * `B.2.3.4`
   */
  virtual void constrained_param_names(std::vector<std::string>& param_names,
                                       bool include_tparams = true,
                                       bool include_gqs = true) const = 0;

  /**
   * Set the specified sequence of parameter names to the
   * unconstrained parameter names.  Each unconstrained parameter is
   * represented as a simple one-dimensional sequence of values.  The
   * actual transforms are documented in the reference manual.
   *
   * <p>The sizes will not be the declared sizes for types such as
   * simplexes, correlation, and covariance matrices.  A simplex of
   * size `N` has `N - 1` unconstrained parameters, an `N` by `N`
   * correlation matrix (or Cholesky factor thereof) has `N` choose 2
   * unconstrained parameters, and a covariance matrix (or Cholesky
   * factor thereof) has `N` choose 2 plus `N` unconstrained
   * parameters.
   *
   * <p>Full details of the transforms and their underlying
   * representations as sequences are detailed in the Stan reference
   * manual.  This also provides details of the order of each
   * parameter type.
   *
   * @param[in,out] param_names sequence of names to set
   * @param[in] include_tparams true if transformed parameters should
   * be included
   * @param[in] include_gqs true if generated quantities should be
   * included
   */
  virtual void unconstrained_param_names(std::vector<std::string>& param_names,
                                         bool include_tparams = true,
                                         bool include_gqs = true) const = 0;

  /**
   * Return the log density for the specified unconstrained
   * parameters, without Jacobian and with normalizing constants for
   * probability functions.
   *
   * @param[in] params_r unconstrained parameters
   * @param[in,out] msgs message stream
   * @return log density for specified parameters
   */
  virtual double log_prob(Eigen::VectorXd& params_r,
                          std::ostream* msgs) const = 0;

  /**
   * Return the log density for the specified unconstrained
   * parameters, without Jacobian and with normalizing constants for
   * probability functions.
   *
   * @param[in] params_r unconstrained parameters
   * @param[in,out] msgs message stream
   * @return log density for specified parameters
   */
  virtual math::var log_prob(Eigen::Matrix<math::var, -1, 1>& params_r,
                             std::ostream* msgs) const = 0;

  /**
   * Return the log density for the specified unconstrained
   * parameters, with Jacobian correction for constraints and with
   * normalizing constants for probability functions.
   *
   * <p>The Jacobian is of the inverse transform from unconstrained
   * parameters to constrained parameters; full details for Stan
   * language types can be found in the language reference manual.
   *
   * @param[in] params_r unconstrained parameters
   * @param[in,out] msgs message stream
   * @return log density for specified parameters
   */
  virtual double log_prob_jacobian(Eigen::VectorXd& params_r,
                                   std::ostream* msgs) const = 0;

  /**
   * Return the log density for the specified unconstrained
   * parameters, with Jacobian correction for constraints and with
   * normalizing constants for probability functions.
   *
   * <p>The Jacobian is of the inverse transform from unconstrained
   * parameters to constrained parameters; full details for Stan
   * language types can be found in the language reference manual.
   *
   * @param[in] params_r unconstrained parameters
   * @param[in,out] msgs message stream
   * @return log density for specified parameters
   */
  virtual math::var log_prob_jacobian(Eigen::Matrix<math::var, -1, 1>& params_r,
                                      std::ostream* msgs) const = 0;

  /**
   * Return the log density for the specified unconstrained
   * parameters, without Jacobian correction for constraints and
   * dropping normalizing constants.
   *
   * <p>This method is for completeness as `double`-based inputs are
   * always constant and will thus cause all probability functions to
   * be dropped from the result.  To get the value of this
   * calculation, use the overload for `math::var`.
   *
   * @param[in] params_r unconstrained parameters
   * @param[in,out] msgs message stream
   * @return log density for specified parameters
   */
  virtual double log_prob_propto(Eigen::VectorXd& params_r,
                                 std::ostream* msgs) const = 0;

  /**
   * Return the log density for the specified unconstrained
   * parameters, without Jacobian correction for constraints and
   * dropping normalizing constants.
   *
   * @param[in] params_r unconstrained parameters
   * @param[in,out] msgs message stream
   * @return log density for specified parameters
   */
  virtual math::var log_prob_propto(Eigen::Matrix<math::var, -1, 1>& params_r,
                                    std::ostream* msgs) const = 0;

  /**
   * Return the log density for the specified unconstrained
   * parameters, with Jacobian correction for constraints and dropping
   * normalizing constants.
   *
   * <p>The Jacobian is of the inverse transform from unconstrained
   * parameters to constrained parameters; full details for Stan
   * language types can be found in the language reference manual.
   *
   * <p>This method is for completeness as `double`-based inputs are
   * always constant and will thus cause all probability functions to
   * be dropped from the result.  To get the value of this
   * calculation, use the overload for `math::var`.
   *
   * @param[in] params_r unconstrained parameters
   * @param[in,out] msgs message stream
   * @return log density for specified parameters
   */
  virtual double log_prob_propto_jacobian(Eigen::VectorXd& params_r,
                                          std::ostream* msgs) const = 0;

  /**
   * Return the log density for the specified unconstrained
   * parameters, with Jacobian correction for constraints and dropping
   * normalizing constants.
   *
   * <p>The Jacobian is of the inverse transform from unconstrained
   * parameters to constrained parameters; full details for Stan
   * language types can be found in the language reference manual.
   *
   * @param[in] params_r unconstrained parameters
   * @param[in,out] msgs message stream
   * @return log density for specified parameters
   */
  virtual math::var log_prob_propto_jacobian(
      Eigen::Matrix<math::var, -1, 1>& params_r, std::ostream* msgs) const = 0;

  /**
   * Convenience template function returning the log density for the
   * specified unconstrained parameters, with Jacobian and normalizing
   * constant inclusion controlled by the template parameters.
   *
   * <p>This non-virtual template method delegates to the appropriate
   * overloaded virtual function.  This allows external interfaces to
   * call the convenient template methods rather than the individual
   * virtual functions.
   *
   * @tparam propto `true` if normalizing constants should be dropped
   * and result returned up to an additive constant
   * @tparam jacobian `true` if the log Jacobian adjustment is
   * included for the change of variables from unconstrained to
   * constrained parameters
   * @tparam T type of scalars in the vector of parameters
   * @param[in] params_r unconstrained parameters
   * @param[in,out] msgs stream to which messages are written
   * @return log density with normalizing constants and Jacobian
   * included as specified by the template parameters
   */
  template <bool propto, bool jacobian, typename T>
  inline T log_prob(Eigen::Matrix<T, -1, 1>& params_r,
                    std::ostream* msgs) const {
    if (propto && jacobian)
      return log_prob_propto_jacobian(params_r, msgs);
    else if (propto && !jacobian)
      return log_prob_propto(params_r, msgs);
    else if (!propto && jacobian)
      return log_prob_jacobian(params_r, msgs);
    else  // if (!propto && !jacobian)
      return log_prob(params_r, msgs);
  }

  /**
   * Read constrained parameter values from the specified context,
   * unconstrain them, then concatenate the unconstrained sequences
   * into the specified parameter sequence.  Output messages go to the
   * specified stream.
   *
   * @param[in] context definitions of variable values
   * @param[in,out] params_r unconstrained parameter values produced
   * @param[in,out] msgs stream to which messages are written
   */
  virtual void transform_inits(const io::var_context& context,
                               Eigen::VectorXd& params_r,
                               std::ostream* msgs) const = 0;

  /**
   * Convert the specified sequence of unconstrained parameters to a
   * sequence of constrained parameters, optionally including
   * transformed parameters and including generated quantities.  The
   * generated quantities may use the random number generator.  Any
   * messages are written to the specified stream.  The output
   * parameter sequence will be resized if necessary to match the
   * number of constrained scalar parameters.
   *
   * @param base_rng RNG to use for generated quantities
   * @param[in] params_r unconstrained parameters input
   * @param[in,out] params_constrained_r constrained parameters produced
   * @param[in] include_tparams true if transformed parameters are
   * included in output
   * @param[in] include_gqs true if generated quantities are included
   * in output
   * @param[in,out] msgs msgs stream to which messages are written
   */
  virtual void write_array(boost::ecuyer1988& base_rng,
                           Eigen::VectorXd& params_r,
                           Eigen::VectorXd& params_constrained_r,
                           bool include_tparams = true, bool include_gqs = true,
                           std::ostream* msgs = 0) const = 0;

  // TODO(carpenter): cut redundant std::vector versions from here ===

  /**
   * Return the log density for the specified unconstrained
   * parameters, without Jacobian and with normalizing constants for
   * probability functions.
   *
   * \deprecated Use Eigen vector versions
   *
   * @param[in] params_r unconstrained parameters
   * @param[in] params_i integer parameters (ignored)
   * @param[in,out] msgs message stream
   * @return log density for specified parameters
   */
  virtual double log_prob(std::vector<double>& params_r,
                          std::vector<int>& params_i,
                          std::ostream* msgs) const = 0;

  /**
   * Return the log density for the specified unconstrained
   * parameters, without Jacobian and with normalizing constants for
   * probability functions.
   *
   * \deprecated Use Eigen vector versions
   *
   * @param[in] params_r unconstrained
   * @param[in] params_i integer parameters (ignored)
   * @param[in,out] msgs message stream
   * @return log density for specified parameters
   */
  virtual math::var log_prob(std::vector<math::var>& params_r,
                             std::vector<int>& params_i,
                             std::ostream* msgs) const = 0;

  /**
   * Return the log density for the specified unconstrained
   * parameters, with Jacobian correction for constraints and with
   * normalizing constants for probability functions.
   *
   * <p>The Jacobian is of the inverse transform from unconstrained
   * parameters to constrained parameters; full details for Stan
   * language types can be found in the language reference manual.
   *
   * \deprecated Use Eigen vector versions
   *
   * @param[in] params_r unconstrained parameters
   * @param[in] params_i integer parameters (ignored)
   * @param[in,out] msgs message stream
   * @return log density for specified parameters
   */
  virtual double log_prob_jacobian(std::vector<double>& params_r,
                                   std::vector<int>& params_i,
                                   std::ostream* msgs) const = 0;

  /**
   * Return the log density for the specified unconstrained
   * parameters, with Jacobian correction for constraints and with
   * normalizing constants for probability functions.
   *
   * <p>The Jacobian is of the inverse transform from unconstrained
   * parameters to constrained parameters; full details for Stan
   * language types can be found in the language reference manual.
   *
   * \deprecated Use Eigen vector versions
   *
   * @param[in] params_r unconstrained parameters
   * @param[in] params_i integer parameters (ignored)
   * @param[in,out] msgs message stream
   * @return log density for specified parameters
   */
  virtual math::var log_prob_jacobian(std::vector<math::var>& params_r,
                                      std::vector<int>& params_i,
                                      std::ostream* msgs) const = 0;

  /**
   * Return the log density for the specified unconstrained
   * parameters, without Jacobian correction for constraints and
   * dropping normalizing constants.
   *
   * <p>This method is for completeness as `double`-based inputs are
   * always constant and will thus cause all probability functions to
   * be dropped from the result.  To get the value of this
   * calculation, use the overload for `math::var`.
   *
   * \deprecated Use Eigen vector versions
   *
   * @param[in] params_r unconstrained parameters
   * @param[in] params_i integer parameters (ignored)
   * @param[in,out] msgs message stream
   * @return log density for specified parameters
   */
  virtual double log_prob_propto(std::vector<double>& params_r,
                                 std::vector<int>& params_i,
                                 std::ostream* msgs) const = 0;

  /**
   * Return the log density for the specified unconstrained
   * parameters, without Jacobian correction for constraints and
   * dropping normalizing constants.
   *
   * \deprecated Use Eigen vector versions
   *
   * @param[in] params_r unconstrained parameters
   * @param[in] params_i integer parameters (ignored)
   * @param[in,out] msgs message stream
   * @return log density for specified parameters
   */
  virtual math::var log_prob_propto(std::vector<math::var>& params_r,
                                    std::vector<int>& params_i,
                                    std::ostream* msgs) const = 0;

  /**
   * Return the log density for the specified unconstrained
   * parameters, with Jacobian correction for constraints and dropping
   * normalizing constants.
   *
   * <p>The Jacobian is of the inverse transform from unconstrained
   * parameters to constrained parameters; full details for Stan
   * language types can be found in the language reference manual.
   *
   * <p>This method is for completeness as `double`-based inputs are
   * always constant and will thus cause all probability functions to
   * be dropped from the result.  To get the value of this
   * calculation, use the overload for `math::var`.
   *
   * \deprecated Use Eigen vector versions
   *
   * @param[in] params_r unconstrained parameters
   * @param[in] params_i integer parameters (ignored)
   * @param[in,out] msgs message stream
   * @return log density for specified parameters
   */
  virtual double log_prob_propto_jacobian(std::vector<double>& params_r,
                                          std::vector<int>& params_i,
                                          std::ostream* msgs) const = 0;

  /**
   * Return the log density for the specified unconstrained
   * parameters, with Jacobian correction for constraints and dropping
   * normalizing constants.
   *
   * <p>The Jacobian is of the inverse transform from unconstrained
   * parameters to constrained parameters; full details for Stan
   * language types can be found in the language reference manual.
   *
   * \deprecated Use Eigen vector versions
   *
   * @param[in] params_r unconstrained parameters
   * @param[in] params_i integer parameters (ignored)
   * @param[in,out] msgs message stream
   * @return log density for specified parameters
   */
  virtual math::var log_prob_propto_jacobian(std::vector<math::var>& params_r,
                                             std::vector<int>& params_i,
                                             std::ostream* msgs) const = 0;

  /**
   * Convenience template function returning the log density for the
   * specified unconstrained parameters, with Jacobian and normalizing
   * constant inclusion controlled by the template parameters.
   *
   * <p>This non-virtual template method delegates to the appropriate
   * overloaded virtual function.  This allows external interfaces to
   * call the convenient template methods rather than the individual
   * virtual functions.
   *
   * \deprecated Use Eigen vector versions
   *
   * @tparam propto `true` if normalizing constants should be dropped
   * and result returned up to an additive constant
   * @tparam jacobian `true` if the log Jacobian adjustment is
   * included for the change of variables from unconstrained to
   * constrained parameters.
   * @tparam T type of scalars in the vector of parameters
   * @param[in] params_r unconstrained parameters
   * @param[in] params_i integer parameters (ignored)
   * @param[in,out] msgs stream to which messages are written
   * @return log density with normalizing constants and Jacobian
   * included as specified by the template parameters
   */
  template <bool propto, bool jacobian, typename T>
  inline T log_prob(std::vector<T>& params_r, std::vector<int>& params_i,
                    std::ostream* msgs) const {
    if (propto && jacobian)
      return log_prob_propto_jacobian(params_r, params_i, msgs);
    else if (propto && !jacobian)
      return log_prob_propto(params_r, params_i, msgs);
    else if (!propto && jacobian)
      return log_prob_jacobian(params_r, params_i, msgs);
    else  // if (!propto && !jacobian)
      return log_prob(params_r, params_i, msgs);
  }

  /**
   * Read constrained parameter values from the specified context,
   * unconstrain them, then concatenate the unconstrained sequences
   * into the specified parameter sequence.  Output messages go to the
   * specified stream.
   *
   * \deprecated Use Eigen vector versions
   *
   * @param[in] context definitions of variable values
   * @param[in] params_i integer parameters (ignored)
   * @param[in,out] params_r unconstrained parameter values produced
   * @param[in,out] msgs stream to which messages are written
   */
  virtual void transform_inits(const io::var_context& context,
                               std::vector<int>& params_i,
                               std::vector<double>& params_r,
                               std::ostream* msgs) const = 0;

  /**
   * Convert the specified sequence of unconstrained parameters to a
   * sequence of constrained parameters, optionally including
   * transformed parameters and including generated quantities.  The
   * generated quantities may use the random number generator.  Any
   * messages are written to the specified stream.  The output
   * parameter sequence will be resized if necessary to match the
   * number of constrained scalar parameters.
   *
   * @param base_rng RNG to use for generated quantities
   * @param[in] params_r unconstrained parameters input
   * @param[in] params_i integer parameters (ignored)
   * @param[in,out] params_r_constrained constrained parameters produced
   * @param[in] include_tparams true if transformed parameters are
   * included in output
   * @param[in] include_gqs true if generated quantities are included
   * in output
   * @param[in,out] msgs msgs stream to which messages are written
   */
  virtual void write_array(boost::ecuyer1988& base_rng,
                           std::vector<double>& params_r,
                           std::vector<int>& params_i,
                           std::vector<double>& params_r_constrained,
                           bool include_tparams = true, bool include_gqs = true,
                           std::ostream* msgs = 0) const = 0;

#ifdef STAN_MODEL_FVAR_VAR

  /**
   * Return the log density for the specified unconstrained
   * parameters, without Jacobian and with normalizing constants for
   * probability functions.
   *
   * @param[in] params_r unconstrained parameters
   * @param[in,out] msgs message stream
   * @return log density for specified parameters
   */
  virtual math::fvar<math::var> log_prob(
      Eigen::Matrix<math::fvar<math::var>, -1, 1>& params_r,
      std::ostream* msgs) const = 0;

  /**
   * Return the log density for the specified unconstrained
   * parameters, with Jacobian correction for constraints and with
   * normalizing constants for probability functions.
   *
   * <p>The Jacobian is of the inverse transform from unconstrained
   * parameters to constrained parameters; full details for Stan
   * language types can be found in the language reference manual.
   *
   * @param[in] params_r unconstrained parameters
   * @param[in,out] msgs message stream
   * @return log density for specified parameters
   */
  virtual math::fvar<math::var> log_prob_jacobian(
      Eigen::Matrix<math::fvar<math::var>, -1, 1>& params_r,
      std::ostream* msgs) const = 0;

  /**
   * Return the log density for the specified unconstrained
   * parameters, without Jacobian correction for constraints and
   * dropping normalizing constants.
   *
   * @param[in] params_r unconstrained parameters
   * @param[in,out] msgs message stream
   * @return log density for specified parameters
   */
  virtual math::fvar<math::var> log_prob_propto(
      Eigen::Matrix<math::fvar<math::var>, -1, 1>& params_r,
      std::ostream* msgs) const = 0;

  /**
   * Return the log density for the specified unconstrained
   * parameters, with Jacobian correction for constraints and dropping
   * normalizing constants.
   *
   * <p>The Jacobian is of the inverse transform from unconstrained
   * parameters to constrained parameters; full details for Stan
   * language types can be found in the language reference manual.
   *
   * @param[in] params_r unconstrained parameters
   * @param[in,out] msgs message stream
   * @return log density for specified parameters
   */
  virtual math::fvar<math::var> log_prob_propto_jacobian(
      Eigen::Matrix<math::fvar<math::var>, -1, 1>& params_r,
      std::ostream* msgs) const = 0;
#endif
};

}  // namespace model
}  // namespace stan
#endif
