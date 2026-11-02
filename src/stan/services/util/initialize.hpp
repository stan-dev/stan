#ifndef STAN_SERVICES_UTIL_INITIALIZE_HPP
#define STAN_SERVICES_UTIL_INITIALIZE_HPP

#include <stan/callbacks/logger.hpp>
#include <stan/callbacks/writer.hpp>
#include <stan/io/var_context.hpp>
#include <stan/io/random_var_context.hpp>
#include <stan/io/chained_var_context.hpp>
#include <stan/model/log_prob_grad.hpp>
#include <stan/math/prim.hpp>
#include <chrono>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace stan {
namespace services {
namespace util {
namespace internal {

/**
 * Returns the unconstrained parameters for a partial initialization.
 *
 * The parameters that <code>init</code> provides keep their constrained
 * values.  The other parameters take the random unconstrained values of
 * <code>random_context</code>.
 *
 * The unconstrained values of the provided parameters are computed by a
 * fixed-point iteration on flat parameter vectors in the order of
 * <code>write_array</code>.  Each pass constrains the current
 * unconstrained values with <code>write_array</code>, replaces the
 * provided parameters with their values from <code>init</code>, frees the
 * result with <code>unconstrain_array</code>, and resets the missing
 * parameters to the random values.  A transform uses only parameters
 * declared before it, so the iteration stops after at most one pass per
 * parameter.
 *
 * The values of the provided parameters in this order come from
 * <code>transform_inits</code> with <code>init</code> and
 * <code>random_context</code>, followed by <code>write_array</code>.  They
 * can differ from <code>init</code> by rounding, or by the projection onto
 * the constraint (for example a simplex).  If a value of
 * <code>random_context</code> violates a bound that depends on a provided
 * value (for example <code>lower=a</code> with a provided <code>a</code>),
 * <code>transform_inits</code> throws and the caller rejects the attempt.
 *
 * A parameter counts as provided if <code>init</code> has any of its
 * elements (a tuple parameter has one name per element).  If the
 * parameter names do not match the parameter vectors, the result is that
 * of <code>transform_inits</code> with <code>init</code> and
 * <code>random_context</code>.  This occurs only for hand-written models;
 * stanc generates the names and the vectors from the same declarations.
 *
 * @tparam Model the type of the model class
 * @tparam InitContext the type of the user-provided var_context
 * @tparam RNG the type of the random number generator
 * @param[in] model the model
 * @param[in] init a var_context with initial values for some parameters
 * @param[in] random_context random initial values for all parameters
 * @param[in,out] rng random number generator (<code>write_array</code>
 *   is called without generated quantities and does not draw from it)
 * @param[in,out] msg stream for messages from the model
 * @throws std::domain_error if a transform or a constraint check fails;
 *   other exceptions from the model are passed through
 * @return unconstrained parameters
 */
template <typename Model, typename InitContext, typename RNG>
std::vector<double> partial_init_unconstrained(
    Model& model, const InitContext& init,
    const stan::io::random_var_context& random_context, RNG& rng,
    std::stringstream& msg) {
  std::vector<std::string> param_names;
  model.get_param_names(param_names, false, false);

  // Tuple elements are "t.1" in the parameter names and "t:1" in the
  // flat parameter names.
  auto base_name = [](const std::string& name) {
    return name.substr(0, name.find_first_of(".:"));
  };
  std::set<std::string> all_bases;
  std::set<std::string> provided;
  for (const auto& name : param_names) {
    all_bases.insert(base_name(name));
    if (init.contains_r(name)) {
      provided.insert(base_name(name));
    }
  }

  std::vector<int> disc_vector;
  std::vector<double> unconstrained;
  stan::io::chained_var_context first_context(init, random_context);
  model.transform_inits(first_context, disc_vector, unconstrained, &msg);
  std::vector<double> provided_values;
  model.write_array(rng, unconstrained, disc_vector, provided_values, false,
                    false, &msg);

  std::vector<std::string> unconstrained_names;
  model.unconstrained_param_names(unconstrained_names, false, false);
  std::vector<std::string> constrained_names;
  model.constrained_param_names(constrained_names, false, false);
  const std::vector<double> u_random = random_context.get_unconstrained();
  if (unconstrained_names.size() != u_random.size()
      || unconstrained_names.size() != unconstrained.size()
      || constrained_names.size() != provided_values.size()) {
    return unconstrained;
  }
  auto is_missing = [&](const std::string& name) {
    const std::string base = base_name(name);
    return all_bases.count(base) > 0 && provided.count(base) == 0;
  };
  std::vector<size_t> missing_u;
  for (size_t n = 0; n < unconstrained_names.size(); ++n) {
    if (is_missing(unconstrained_names[n])) {
      missing_u.push_back(n);
    }
  }
  std::vector<size_t> provided_c;
  for (size_t n = 0; n < constrained_names.size(); ++n) {
    if (!is_missing(constrained_names[n])) {
      provided_c.push_back(n);
    }
  }

  for (size_t n : missing_u) {
    unconstrained[n] = u_random[n];
  }
  std::vector<double> constrained;
  std::vector<double> next;
  // Without convergence (for example a NaN in init) the last pass is
  // returned; the log density check of the caller usually rejects it.
  for (size_t pass = 0; pass <= param_names.size(); ++pass) {
    model.write_array(rng, unconstrained, disc_vector, constrained, false,
                      false, &msg);
    for (size_t n : provided_c) {
      constrained[n] = provided_values[n];
    }
    model.unconstrain_array(constrained, next, &msg);
    for (size_t n : missing_u) {
      next[n] = u_random[n];
    }
    if (next == unconstrained) {
      break;
    }
    std::swap(unconstrained, next);
  }
  return unconstrained;
}

}  // namespace internal

/**
 * Returns a valid initial value of the parameters of the model
 * on the unconstrained scale.
 *
 * For identical inputs (model, init, rng, init_radius), this
 * function will produce the same initialization.
 *
 * Initialization first tries to use the provided
 * <code>stan::io::var_context</code>, then it will generate
 * random uniform values from -init_radius to +init_radius on the
 * unconstrained scale for missing parameters.  The provided values are
 * kept on the constrained scale, also when an offset or multiplier of a
 * missing parameter depends on a provided parameter or the reverse.  An
 * attempt is still rejected if the random constrained value of a missing
 * parameter violates a bound that depends on a provided value.
 *
 * When the <code>var_context</code> provides all variables or
 * the init_radius is 0, this function will only evaluate the
 * log probability of the model with the unconstrained
 * parameters once to see if it's valid.
 *
 * When at least some of the initialization is random, it will
 * randomly initialize until it finds a set of unconstrained
 * parameters that are valid or it hits <code>max_tries</code>,
 * which defaults to 100.
 *
 * Valid initialization is defined as a finite, non-NaN value for the
 * evaluation of the log probability density function and all its
 * gradients.
 *
 * @tparam Jacobian indicates whether to include the Jacobian term when
 *   evaluating the log density function
 * @tparam Model the type of the model class
 * @tparam RNG the type of the random number generator
 *
 * @param[in] model the model
 * @param[in] init a var_context with initial values
 * @param[in,out] rng random number generator
 * @param[in] init_radius the radius for generating random values.
 *   A value of 0 indicates that the unconstrained parameters (not
 *   provided by init) should be initialized with 0.
 * @param[in] print_timing indicates whether a timing message should
 *   be printed to the logger
 * @param[in,out] logger logger for messages
 * @param[in,out] init_writer init writer (on the unconstrained scale)
 * @param[in] max_tries The maximum number of times a random initialization
 *   will be re-tried to achieve a finite log density and gradient. Default 100.
 * @throws exception passed through from the model if the model has a
 *   fatal error (not a std::domain_error)
 * @throws std::domain_error if the model can not be initialized and
 *   the model does not have a fatal error (only allows for
 *   std::domain_error)
 * @return valid unconstrained parameters for the model
 */
template <bool Jacobian = true, typename Model, typename InitContext,
          typename RNG>
std::vector<double> initialize(Model& model, const InitContext& init, RNG& rng,
                               double init_radius, bool print_timing,
                               stan::callbacks::logger& logger,
                               stan::callbacks::writer& init_writer,
                               int max_tries = 100) {
  std::vector<double> unconstrained;
  std::vector<int> disc_vector;

  bool is_fully_initialized = true;
  bool any_initialized = false;
  std::vector<std::string> param_names;
  model.get_param_names(param_names, false, false);
  for (size_t n = 0; n < param_names.size(); n++) {
    is_fully_initialized &= init.contains_r(param_names[n]);
    any_initialized |= init.contains_r(param_names[n]);
  }

  bool is_initialized_with_zero = init_radius == 0.0;

  int MAX_INIT_TRIES
      = is_fully_initialized || is_initialized_with_zero ? 1 : max_tries;
  int num_init_tries = 0;
  for (; num_init_tries < MAX_INIT_TRIES; num_init_tries++) {
    std::stringstream msg;
    try {
      stan::io::random_var_context random_context(model, rng, init_radius,
                                                  is_initialized_with_zero);

      if (!any_initialized) {
        unconstrained = random_context.get_unconstrained();
      } else if (is_fully_initialized) {
        stan::io::chained_var_context context(init, random_context);

        model.transform_inits(context, disc_vector, unconstrained, &msg);
      } else {
        unconstrained = internal::partial_init_unconstrained(
            model, init, random_context, rng, msg);
      }
    } catch (std::domain_error& e) {
      if (msg.str().length() > 0) {
        logger.info(msg);
      }
      logger.warn("Rejecting initial value:");
      logger.warn(
          "  Error evaluating the log probability"
          " at the initial value.");
      logger.warn(e.what());
      continue;
    } catch (std::exception& e) {
      if (msg.str().length() > 0) {
        logger.info(msg);
      }
      logger.error(
          "Unrecoverable error evaluating the log probability"
          " at the initial value.");
      throw;
    }

    msg.str("");
    double log_prob(0);
    try {
      // we evaluate the log_prob function with propto=false
      // because we're evaluating with `double` as the type of
      // the parameters.
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
          "  Error evaluating the log probability"
          " at the initial value.");
      logger.warn(e.what());
      continue;
    } catch (std::exception& e) {
      if (msg.str().length() > 0) {
        logger.info(msg);
      }
      logger.error(
          "Unrecoverable error evaluating the log probability"
          " at the initial value.");
      throw;
    }
    if (!std::isfinite(log_prob)) {
      logger.warn("Rejecting initial value:");
      logger.warn(
          "  Log probability evaluates to log(0),"
          " i.e. negative infinity.");
      logger.warn(
          "  Stan can't start sampling from this"
          " initial value.");
      continue;
    }
    std::stringstream log_prob_msg;
    std::vector<double> gradient;
    auto start = std::chrono::steady_clock::now();
    try {
      // we evaluate this with propto=true since we're
      // evaluating with autodiff variables
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
      logger.warn(
          "  Gradient evaluated at the initial value"
          " is not finite.");
      logger.warn(
          "  Stan can't start sampling from this"
          " initial value.");
    }
    if (gradient_ok && print_timing) {
      logger.info("");
      std::stringstream msg1;
      msg1 << "Gradient evaluation took " << deltaT << " seconds";
      logger.info(msg1);

      std::stringstream msg2;
      msg2 << "1000 transitions using 10 leapfrog steps"
           << " per transition would take"
           << " " << 1e4 * deltaT << " seconds.";
      logger.info(msg2);

      logger.info("Adjust your expectations accordingly!");
      logger.info("");
      logger.info("");
    }
    if (gradient_ok) {
      init_writer(unconstrained);
      return unconstrained;
    }
  }
  if (is_fully_initialized) {
    logger.info("");
    logger.error("User-specified initialization failed.");
    logger.error(
        " Try specifying new initial values,"
        " using partially specialized initialization,"
        " reducing the range of constrained values,"
        " or reparameterizing the model.");
  } else if (any_initialized) {
    logger.info("");
    std::stringstream msg;
    msg << "Partial user-specified initialization failed. "
           "Initialization of non user specified parameters "
           "between (-"
        << init_radius << ", " << init_radius << ") failed after"
        << " " << MAX_INIT_TRIES << " attempts. ";
    logger.error(msg);
    logger.error(
        " Try specifying full initial values,"
        " reducing the range of constrained values,"
        " or reparameterizing the model.");
  } else if (is_initialized_with_zero) {
    logger.info("");
    logger.error("Initial values of 0 failed to initialize.");
    logger.error(
        " Try specifying new initial values,"
        " using partially specialized initialization,"
        " reducing the range of constrained values,"
        " or reparameterizing the model.");
  } else {
    logger.info("");
    std::stringstream msg;
    msg << "Initialization between (-" << init_radius << ", " << init_radius
        << ") failed after"
        << " " << MAX_INIT_TRIES << " attempts. ";
    logger.error(msg);
    logger.error(
        " Try specifying initial values,"
        " reducing ranges of constrained values,"
        " or reparameterizing the model.");
  }
  throw std::domain_error("Initialization failed.");
}

}  // namespace util
}  // namespace services
}  // namespace stan
#endif
