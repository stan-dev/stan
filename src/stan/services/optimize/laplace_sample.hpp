#ifndef STAN_SERVICES_OPTIMIZE_LAPLACE_SAMPLE_HPP
#define STAN_SERVICES_OPTIMIZE_LAPLACE_SAMPLE_HPP

#include <stan/callbacks/interrupt.hpp>
#include <stan/callbacks/logger.hpp>
#include <stan/callbacks/writer.hpp>
#include <stan/math/rev.hpp>
#include <stan/services/error_codes.hpp>
#include <stan/services/util/create_rng.hpp>
#include <string>
#include <vector>

namespace stan {
namespace services {

/**
 * Take the specified number of draws from the Laplace approximation
 * for the model at the specified unconstrained mode, writing the
 * draws, unnormalized log density, and unnormalized density of the
 * approximation to the sample writer and writing messages to the
 * logger, returning a return code of 0 if successful.
 *
 * Interrupts are called 
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
 * given 
 * @param[in] interrupt callback for interrupting sampling
 * @param[in,out] logger callback for writing console messages from
 * sampler and from Stan programs
 * @param[in,out] sample_writer callback for writing parameter names
 * and then draws
 * @return a return code, with 0 indicating success and
 * `stan::error_codes::DATAERROR` indicating misconfiguration
 * @throw any exception raised in executing the Stan program
 */
template <bool jacobian, typename Model>
int laplace_sample(const Model& model, const Eigen::VectorXd& theta_hat,
		   int draws, unsigned int random_seed,
		   int refresh, callbacks::interrupt& interrupt,
		   callbacks::logger& logger,
		   callbacks::writer& sample_writer) {
  // validate number of draws >= 0
  if (draws <= 0) {
    std::stringstream msg;
    msg << "Number of draws must be > 0, found " << draws << std::endl;
    std::string msg_str = msg.str();
    logger.error(msg_str);
    return error_codes::DATAERR;
  }

  // validate mode is correct size
  std::vector<std::string> unc_param_names;
  model.unconstrained_param_names(unc_param_names, false, false);
  int num_unc_params = unc_param_names.size();
  if (theta_hat.size() != num_unc_params) {
    std::stringstream msg;
    msg << "Specified mode is wrong size; expected "
	<< num_unc_params << " unconstrained parameters"
	<< ", but specified mode has size " << theta_hat.size()
	<< std::endl;
    std::string msg_str = msg.str();
    logger.error(msg_str);
    return error_codes::DATAERR;
  }

  std::vector<std::string> param_tp_gq_names;
  model.constrained_param_names(param_tp_gq_names, true, true);
  size_t draw_size = param_tp_gq_names.size();

  // write names of params, tps, and gqs to sample writer
  std::vector<std::string> names;
  static const bool include_tp = true;
  static const bool include_gq = true;
  model.constrained_param_names(names, include_tp, include_gq);
  names.push_back("log_p");
  names.push_back("log_q");
  sample_writer(names);

  // create log density functor
  std::stringstream log_density_msgs;
  auto log_density_fun
    = [&](const Eigen::Matrix<stan::math::var, -1, 1>& theta) {
    return model.template log_prob<true, jacobian, stan::math::var>(
      const_cast<Eigen::Matrix<stan::math::var, -1, 1>&>(theta),
      &log_density_msgs);
  };

  // calculate inverse negative Hessian's Cholesky factor
  logger.info("Calculating Hessian\n");
  double log_p;  // dummy
  Eigen::VectorXd grad;  // dummy
  Eigen::MatrixXd hessian;
  interrupt();
  math::internal::finite_diff_hessian_auto(log_density_fun, theta_hat, log_p,
					   grad, hessian);
  std::string log_density_msgs_str = log_density_msgs.str();
  logger.info(log_density_msgs_str);

  // calculate Cholesky factor and inverse
  interrupt();
  logger.info("\nCalculating inverse of Cholesky factor\n");
  Eigen::MatrixXd L_neg_hessian = (-hessian).llt().matrixL();
  interrupt();
  Eigen::MatrixXd inv_sqrt_neg_hessian = L_neg_hessian.inverse();
  interrupt();
  Eigen::MatrixXd neg_half_inv = -0.5 * inv_sqrt_neg_hessian * inv_sqrt_neg_hessian.transpose();
  // don't need log-determinant because log_q is unnormalized
  // double log_det = 2 * L_neg_hessian.diagonal().array().log().sum();

  // generate draws
  interrupt();
  logger.info("\nGenerating draws\n");
  boost::ecuyer1988 rng = util::create_rng(random_seed, 0);
  Eigen::VectorXd draw_vec;  // declare draw_vec, msgs here to avoid re-alloc
  for (int m = 0; m < draws; ++m) {
    interrupt();  // allow interpution each iteration
    if (refresh > 0 && m % refresh == 0) {
      std::stringstream log_msg;
      log_msg << "iteration " << m << std::endl;
      logger.info(log_msg);
    }
    Eigen::VectorXd z(num_unc_params);
    for (int n = 0; n < num_unc_params; ++n) {
      z(n) = math::std_normal_rng(rng);
    }

    Eigen::VectorXd unc_draw = theta_hat + inv_sqrt_neg_hessian * z;
    double log_p = 1.0;
    // double log_p = log_density_fun(unc_draw);
    Eigen::VectorXd diff = unc_draw - theta_hat;
    double log_q = diff.transpose() * neg_half_inv * diff;
    
    std::stringstream msgs;
    model.write_array(rng, unc_draw, draw_vec, include_tp, include_gq, &msgs);
    logger.info(msgs);
    std::vector<double> draw(&draw_vec(0), &draw_vec(0) + draw_size);
    draw.push_back(log_p);
    draw.push_back(log_q);
    sample_writer(draw);
  }
  return error_codes::OK;
}
  
}  // namespace services
}  // namespace stan
  
#endif
