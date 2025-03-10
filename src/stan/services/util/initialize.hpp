#ifndef STAN_SERVICES_UTIL_INITIALIZE_HPP
#define STAN_SERVICES_UTIL_INITIALIZE_HPP

#include <stan/callbacks/logger.hpp>
#include <stan/callbacks/dispatcher.hpp>
#include <stan/io/var_context.hpp>
#include <stan/io/random_var_context.hpp>
#include <stan/io/chained_var_context.hpp>
#include <stan/model/log_prob_grad.hpp>
#include <stan/math/prim.hpp>
#include <chrono>
#include <sstream>
#include <string>
#include <vector>

namespace stan {
namespace services {
namespace util {

/**
 * Checks if a model is fully or partially initialized from a var_context.
 *
 * @tparam Model Type of model
 * @tparam InitContext Type of initialization context
 * @param[in] model Model to check
 * @param[in] init Initialization context
 * @param[out] is_fully_initialized Set to true if all parameters are
 * initialized
 * @param[out] any_initialized Set to true if any parameters are initialized
 */
template <typename Model, typename InitContext>
void check_initialization_status(Model& model, const InitContext& init,
                                 bool& is_fully_initialized,
                                 bool& any_initialized) {
  is_fully_initialized = true;
  any_initialized = false;

  std::vector<std::string> param_names;
  model.get_param_names(param_names, false, false);

  for (size_t n = 0; n < param_names.size(); n++) {
    is_fully_initialized &= init.contains_r(param_names[n]);
    any_initialized |= init.contains_r(param_names[n]);
  }
}

/**
 * Transforms initial values from the constrained to the unconstrained space.
 *
 * @tparam Model Type of model
 * @tparam InitContext Type of initialization context
 * @tparam RNG Type of random number generator
 * @param[in] model Model to initialize
 * @param[in] init Initial values context
 * @param[in] rng Random number generator
 * @param[in] init_radius Radius for random initialization
 * @param[in] is_initialized_with_zero Whether to initialize with zeros
 * @param[in] any_initialized Whether any parameters were initialized
 * @param[out] unconstrained Output vector of unconstrained values
 * @param[out] disc_vector Output vector of discrete parameters
 * @param[in,out] logger Logger for messages
 * @return True if initialization succeeded, false otherwise
 */
template <typename Model, typename InitContext, typename RNG>
bool get_unconstrained_values(Model& model, const InitContext& init, RNG& rng,
                              double init_radius, bool is_initialized_with_zero,
                              bool any_initialized,
                              std::vector<double>& unconstrained,
                              std::vector<int>& disc_vector,
                              stan::callbacks::logger& logger) {
  std::stringstream msg;
  try {
    stan::io::random_var_context random_context(model, rng, init_radius,
                                                is_initialized_with_zero);

    if (!any_initialized) {
      unconstrained = random_context.get_unconstrained();
    } else {
      stan::io::chained_var_context context(init, random_context);
      model.transform_inits(context, disc_vector, unconstrained, &msg);
    }
    return true;
  } catch (std::domain_error& e) {
    if (msg.str().length() > 0) {
      logger.info(msg);
    }
    logger.warn("Rejecting initial value:");
    logger.warn("  Error evaluating the log probability at the initial value.");
    logger.warn(e.what());
    return false;
  } catch (std::exception& e) {
    if (msg.str().length() > 0) {
      logger.info(msg);
    }
    logger.error(
        "Unrecoverable error evaluating the log probability at the initial "
        "value.");
    throw;
  }
}

/**
 * Logs initialization failure messages.
 *
 * @param[in] is_fully_initialized Whether all parameters were initialized
 * @param[in] any_initialized Whether any parameters were initialized
 * @param[in] is_initialized_with_zero Whether initialization was with zeros
 * @param[in] init_radius Radius for random initialization
 * @param[in] max_init_tries Maximum number of initialization attempts
 * @param[in,out] logger Logger for messages
 */
void log_initialization_failure(bool is_fully_initialized, bool any_initialized,
                                bool is_initialized_with_zero,
                                double init_radius, int max_init_tries,
                                stan::callbacks::logger& logger) {
  logger.info("");

  if (is_fully_initialized) {
    logger.error("User-specified initialization failed.");
    logger.error(
        " Try specifying new initial values,"
        " using partially specialized initialization,"
        " reducing the range of constrained values,"
        " or reparameterizing the model.");
  } else if (any_initialized) {
    std::stringstream msg;
    msg << "Partial user-specified initialization failed. "
           "Initialization of non user specified parameters between (-"
        << init_radius << ", " << init_radius << ") failed after"
        << " " << max_init_tries << " attempts. ";
    logger.error(msg);
    logger.error(
        " Try specifying full initial values,"
        " reducing the range of constrained values,"
        " or reparameterizing the model.");
  } else if (is_initialized_with_zero) {
    logger.error("Initial values of 0 failed to initialize.");
    logger.error(
        " Try specifying new initial values,"
        " using partially specialized initialization,"
        " reducing the range of constrained values,"
        " or reparameterizing the model.");
  } else {
    std::stringstream msg;
    msg << "Initialization between (-" << init_radius << ", " << init_radius
        << ") failed after" << " " << max_init_tries << " attempts. ";
    logger.error(msg);
    logger.error(
        " Try specifying initial values,"
        " reducing ranges of constrained values,"
        " or reparameterizing the model.");
  }
}

/**
 * Returns a valid initial value of the parameters of the model
 * on the unconstrained scale.
 *
 * @tparam Jacobian indicates whether to include the Jacobian term
 * @tparam Model the type of the model class
 * @tparam InitContext type of the initialization context
 * @tparam RNG the type of the random number generator
 *
 * @param[in] model the model
 * @param[in] init a var_context with initial values
 * @param[in,out] rng random number generator
 * @param[in] init_radius the radius for generating random values
 * @param[in] print_timing whether to print timing information
 * @param[in,out] logger logger for messages
 * @param[in,out] dispatcher dispatcher for outputs
 * @throws exception from model or std::domain_error on initialization failure
 * @return valid unconstrained parameters for the model
 */
template <bool Jacobian = true, typename Model, typename InitContext,
          typename RNG>
std::vector<double> initialize(Model& model, const InitContext& init, RNG& rng,
                               double init_radius, bool print_timing,
                               stan::callbacks::logger& logger,
                               stan::callbacks::dispatcher& dispatcher) {
  std::vector<double> unconstrained;
  std::vector<int> disc_vector;

  bool is_fully_initialized = false;
  bool any_initialized = false;
  check_initialization_status(model, init, is_fully_initialized,
                              any_initialized);

  bool is_initialized_with_zero = init_radius == 0.0;

  int MAX_INIT_TRIES
      = is_fully_initialized || is_initialized_with_zero ? 1 : 100;

  for (int num_init_tries = 0; num_init_tries < MAX_INIT_TRIES;
       num_init_tries++) {
    // Get unconstrained initial values
    if (!get_unconstrained_values(model, init, rng, init_radius,
                                  is_initialized_with_zero, any_initialized,
                                  unconstrained, disc_vector, logger)) {
      continue;  // Try again with new random values
    }

    // Evaluate log probability - using direct model interface
    std::stringstream msg;
    double log_prob(0);
    try {
      log_prob = model.template log_prob<false, Jacobian>(unconstrained,
                                                          disc_vector, &msg);
      if (msg.str().length() > 0) {
        logger.info(msg);
      }
    } catch (std::domain_error& e) {
      if (msg.str().length() > 0)
        logger.info(msg);
      logger.warn("Rejecting initial value:");
      logger.warn(
          "  Error evaluating the log probability at the initial value.");
      logger.warn(e.what());
      continue;
    } catch (std::exception& e) {
      if (msg.str().length() > 0) {
        logger.info(msg);
      }
      logger.error(
          "Unrecoverable error evaluating the log probability at the initial "
          "value.");
      throw;
    }

    if (!std::isfinite(log_prob)) {
      logger.warn("Rejecting initial value:");
      logger.warn(
          "  Log probability evaluates to log(0), i.e. negative infinity.");
      logger.warn("  Stan can't start sampling from this initial value.");
      continue;  // Try again with new random values
    }

    // Evaluate gradient - using direct model interface
    std::stringstream log_prob_msg;
    std::vector<double> gradient;
    auto start = std::chrono::steady_clock::now();
    try {
      // Use model's log_prob_grad method directly
      log_prob = stan::model::log_prob_grad<true, Jacobian>(
          model, unconstrained, disc_vector, gradient, &log_prob_msg);
    } catch (const std::exception& e) {
      if (log_prob_msg.str().length() > 0) {
        logger.info(log_prob_msg);
      }
      logger.error(e.what());
      throw;
    }
    auto end = std::chrono::steady_clock::now();
    double deltaT
        = std::chrono::duration_cast<std::chrono::microseconds>(end - start)
              .count()
          / 1000000.0;

    if (log_prob_msg.str().length() > 0)
      logger.info(log_prob_msg);

    bool gradient_ok = std::isfinite(stan::math::sum(gradient));

    if (!gradient_ok) {
      logger.warn("Rejecting initial value:");
      logger.warn("  Gradient evaluated at the initial value is not finite.");
      logger.warn("  Stan can't start sampling from this initial value.");
      continue;
    }

    if (gradient_ok && print_timing) {
      logger.info("");
      std::stringstream msg;
      msg << "Gradient evaluation took " << deltaT << " seconds" << std::endl
          << "1000 transitions using 10 leapfrog steps per transition would "
             "take"
          << " " << 1e4 * deltaT << " seconds." << std::endl
          << "Adjust your expectations accordingly!" << std::endl
          << std::endl
          << std::endl;
      logger.info(msg);
    }

    dispatcher.dispatch(stan::callbacks::info_type::UNCONSTRAINED_INITS,
                        unconstrained);
    return unconstrained;
  }

  // If we get here, initialization failed
  log_initialization_failure(is_fully_initialized, any_initialized,
                             is_initialized_with_zero, init_radius,
                             MAX_INIT_TRIES, logger);

  throw std::domain_error("Initialization failed.");
}

// Overload for backward compatibility
template <bool Jacobian = true, typename Model, typename InitContext,
          typename RNG>
std::vector<double> initialize(Model& model, const InitContext& init, RNG& rng,
                               double init_radius, bool print_timing,
                               stan::callbacks::logger& logger,
                               stan::callbacks::writer& init_writer) {
  // Create a temporary dispatcher just for this function
  stan::callbacks::dispatcher dispatcher;

  // Register the init_writer with the dispatcher
  auto channel
      = std::make_unique<stan::callbacks::writer_channel>(&init_writer);
  dispatcher.register_channel(stan::callbacks::info_type::UNCONSTRAINED_INITS,
                              std::move(channel));

  // Call the dispatcher version
  return initialize<Jacobian>(model, init, rng, init_radius, print_timing,
                              logger, dispatcher);
}

}  // namespace util
}  // namespace services
}  // namespace stan
#endif
