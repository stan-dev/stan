#ifndef STAN_SERVICES_OPTIMIZE_LAPLACE_SAMPLE_HPP
#define STAN_SERVICES_OPTIMIZE_LAPLACE_SAMPLE_HPP

#include <stan/callbacks/interrupt.hpp>
#include <stan/callbacks/logger.hpp>
#include <stan/callbacks/writer.hpp>
#include <stan/math/rev.hpp>
#include <stan/services/error_codes.hpp>
#include <stan/services/util/create_rng.hpp>
#include <string>
#include <type_traits>
#include <vector>

namespace stan {
namespace services {
namespace internal {

template <bool jacobian, typename Model>
void laplace_sample(const Model& model, const Eigen::VectorXd& theta_hat,
                    int draws, unsigned int random_seed, int refresh,
                    callbacks::interrupt& interrupt, callbacks::logger& logger,
                    callbacks::writer& sample_writer) {
  if (draws <= 0) {
    throw std::domain_error("Number of draws must be > 0; found draws = "
                            + std::to_string(draws));
  }

  std::vector<std::string> unc_param_names;
  model.unconstrained_param_names(unc_param_names, false, false);
  int num_unc_params = unc_param_names.size();

  if (theta_hat.size() != num_unc_params) {
    throw ::std::domain_error(
        "Specified mode is wrong size; expected "
        + std::to_string(num_unc_params)
        + " unconstrained parameters, but specified mode has size = "
        + std::to_string(theta_hat.size()));
  }

  std::vector<std::string> param_tp_gq_names;
  model.constrained_param_names(param_tp_gq_names, true, true);
  size_t draw_size = param_tp_gq_names.size();

  // write names of params, tps, and gqs to sample writer
  std::vector<std::string> names;
  names.push_back("log_p__");
  names.push_back("log_q__");
  static const bool include_tp = true;
  static const bool include_gq = true;
  model.constrained_param_names(names, include_tp, include_gq);
  sample_writer(names);

  // create log density functor for vars and vals
  std::stringstream log_density_msgs;
  auto log_density_fun
      = [&](const Eigen::Matrix<stan::math::var, -1, 1>& theta) {
          return model.template log_prob<true, jacobian, stan::math::var>(
              const_cast<Eigen::Matrix<stan::math::var, -1, 1>&>(theta),
              &log_density_msgs);
        };

  // calculate inverse negative Hessian's Cholesky factor
  if (refresh > 0) {
    logger.info("Calculating Hessian");
  }
  double log_p;          // dummy
  Eigen::VectorXd grad;  // dummy
  Eigen::MatrixXd hessian;
  interrupt();
  math::internal::finite_diff_hessian_auto(log_density_fun, theta_hat, log_p,
                                           grad, hessian);
  if (refresh > 0 && log_density_msgs.peek() != std::char_traits<char>::eof())
    logger.info(log_density_msgs);

  // calculate Cholesky factor and inverse
  interrupt();
  if (refresh > 0) {
    logger.info("Calculating inverse of Cholesky factor");
  }
  Eigen::MatrixXd L_neg_hessian = (-hessian).llt().matrixL();
  interrupt();
  Eigen::MatrixXd inv_sqrt_neg_hessian = L_neg_hessian.inverse().transpose();
  interrupt();
  Eigen::MatrixXd half_hessian = 0.5 * hessian;

  if (refresh > 0) {
    logger.info("Generating draws");
  }
  // generate draws
  std::stringstream refresh_msg;
  boost::ecuyer1988 rng = util::create_rng(random_seed, 0);
  Eigen::VectorXd draw_vec;  // declare draw_vec, msgs here to avoid re-alloc
  for (int m = 0; m < draws; ++m) {
    interrupt();  // allow interpution each iteration
    if (refresh > 0 && m % refresh == 0) {
      refresh_msg << "iteration: " << std::to_string(m);
      logger.info(refresh_msg);
      refresh_msg.str(std::string());
    }
    Eigen::VectorXd z(num_unc_params);
    for (int n = 0; n < num_unc_params; ++n) {
      z(n) = math::std_normal_rng(rng);
    }
    Eigen::VectorXd unc_draw = theta_hat + inv_sqrt_neg_hessian * z;
    std::stringstream write_array_msgs;
    model.write_array(rng, unc_draw, draw_vec, include_tp, include_gq,
                      &write_array_msgs);
    if (refresh > 0 && write_array_msgs.peek() != std::char_traits<char>::eof())
      logger.info(write_array_msgs);
    // output draw, log_p, log_q
    std::vector<double> draw(&draw_vec(0), &draw_vec(0) + draw_size);
    double log_p = log_density_fun(unc_draw).val();
    draw.insert(draw.begin(), log_p);
    Eigen::VectorXd diff = unc_draw - theta_hat;
    double log_q = diff.transpose() * half_hessian * diff;
    draw.insert(draw.begin() + 1, log_q);
    sample_writer(draw);
  }
}  // namespace internal
}  // namespace internal

/**
 * Take the specified number of draws from the Laplace approximation
 * for the model at the specified unconstrained mode, writing the
 * draws, unnormalized log density, and unnormalized density of the
 * approximation to the sample writer and writing messages to the
 * logger, returning a return code of zero if successful.
 *
 * Interrupts are called between compute-intensive operations.  To
 * turn off all console messages sent to the logger, set refresh to 0.
 * If an exception is thrown by the model, the return value is
 * non-zero, and if refresh > 0, its message is given to the logger as
 * an error.
 *
 * @tparam jacobian `true` to include Jacobian adjustment for
 * constrained parameters
 * @tparam Model a Stan model
 * @parma[in] m model from which to sample
 * @parma[in] theta_hat unconstrained mode at which to center the
 * Laplace approximation
 * @param[in] draws number of draws to generate
 * @param[in] random_seed seed for generating random numbers in the
 * Stan program and in sampling
 * @param[in] refresh period between iterations at which updates are
 * given, with a value of 0 turning off all messages
 * @param[in] interrupt callback for interrupting sampling
 * @param[in,out] logger callback for writing console messages from
 * sampler and from Stan programs
 * @param[in,out] sample_writer callback for writing parameter names
 * and then draws
 * @return a return code, with 0 indicating success
 */
template <bool jacobian, typename Model>
int laplace_sample(const Model& model, const Eigen::VectorXd& theta_hat,
                   int draws, unsigned int random_seed, int refresh,
                   callbacks::interrupt& interrupt, callbacks::logger& logger,
                   callbacks::writer& sample_writer) {
  try {
    internal::laplace_sample<jacobian>(model, theta_hat, draws, random_seed,
                                       refresh, interrupt, logger,
                                       sample_writer);
    return error_codes::OK;
  } catch (const std::exception& e) {
    if (refresh >= 0) {
      logger.error(e.what());
    }
  } catch (...) {
    if (refresh >= 0) {
      logger.error("unknown exception during execution");
    }
  }
  return error_codes::DATAERR;
}
}  // namespace services
}  // namespace stan

#endif
