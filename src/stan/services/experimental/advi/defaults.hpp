#ifndef STAN_SERVICES_DIAGNOSE_DEFAULTS_HPP
#define STAN_SERVICES_DIAGNOSE_DEFAULTS_HPP

#include <stdexcept>
#include <string>

namespace stan {
  namespace services {
    namespace experimental {
      namespace advi {

        /**
         * Number of samples for Monte Carlo estimate of gradients.
         */
        struct gradient_samples {
          /**
           * Description of gradient_samples.
           *
           * @returns description
           */
          static std::string description() {
            return "Number of Monte Carlo draws for computing the gradient.";
          }

          /**
           * Validates gradient_samples; must be greater than 0.
           *
           * @param[in] gradient_samples argument to validate
           * @throw std::invalid_argument if gradient_samples <= 0
           */
          static void validate(int gradient_samples) {
            if (!(gradient_samples > 0))
              throw std::invalid_argument("gradient_samples must be greater "
                                          "than 0.");
          }

          /**
           * Default number of gradient_samples: 1.
           *
           * @returns default
           */
          static int default_value() {
            return 1;
          }
        };

        /**
         * Number of Monte Carlo samples for estimate of ELBO.
         */
        struct elbo_samples {
          /**
           * Description of elbo_samples.
           *
           * @returns description
           */
          static std::string description() {
            return "Number of Monte Carlo draws for estimate of ELBO.";
          }

          /**
           * Validates elbo_samples; must be greater than 0.
           *
           * @param[in] elbo_samples argument to validate
           * @throw std::invalid_argument if elbo_samples <= 0
           */
          static void validate(double elbo_samples) {
            if (!(elbo_samples > 0))
              throw std::invalid_argument("elbo_samples must be greater "
                                          "than 0.");
          }

          /**
           * Default elbo_samples: 100.
           *
           * @returns default
           */
          static int default_value() {
            return 100;
          }
        };

        /**
         * Maximum number of iterations to run ADVI.
         */
        struct max_iterations {
          /**
           * String description of the maximum number of iterations.
           *
           * @returns description
           */
          static std::string description() {
            return "Maximum number of ADVI iterations.";
          }

          /**
           * Validates max_iterations; max_iterations must be greater than 0.
           *
           * @param[in] max_iterations argument to validate
           * @throw std::invalid_argument if max_iterations <= 0
           */
          static void validate(int max_iterations) {
            if (!(max_iterations > 0))
              throw std::invalid_argument("max_iterations must be greater "
                                          "than 0.");
          }

          /**
           * Default max_iterations: 10000.
           *
           * @returns default
           */
          static int default_value() {
            return 10000;
          }
        };


        /**
         * Relative tolerance parameter for convergence.
         */
        struct tol_rel_obj {
          /**
           * Description of tol_rel_obj.
           *
           * @returns description
           */
          static std::string description() {
            return "Relative tolerance parameter for convergence.";
          }

          /**
           * Validates tol_rel_obj; must be greater than 0.
           *
           * @param[in] tol_rel_obj argument to validate
           * @throw std::invalid_argument if tol_rel_obj <= 0
           */
          static void validate(double tol_rel_obj) {
            if (!(tol_rel_obj > 0))
              throw std::invalid_argument("tol_rel_obj must be greater "
                                          "than 0.");
          }

          /**
           * Default tol_rel_obj: 0.01.
           *
           * @returns default
           */
          static double default_value() {
            return 0.01;
          }
        };

        /**
         * Stepsize scaling parameter for variational inference
         */
        struct eta {
          /**
           * Description of eta.
           *
           * @returns description
           */
          static std::string description() {
            return "Stepsize scaling parameter.";
          }

          /**
           * Validates eta; must be greater than 0.
           *
           * @param[in] eta argument to validate
           * @throw std::invalid_argument if eta <= 0
           */
          static void validate(double eta) {
            if (!(eta > 0))
              throw std::invalid_argument("eta must be greater than 0.");
          }

          /**
           * Default.
           *
           * @returns default
           */
          static double default_value() {
            return 1.0;
          }
        };

        /**
         * Flag for eta adaptation.
         */
        struct adapt_engaged {
          /**
           * Description of adapt_engaged.
           *
           * @returns description
           */
          static std::string description() {
            return "Boolean flag for eta adaptation.";
          }

          /**
           * Validates adapt_engaged. This is a noop.
           *
           * @param[in] adapt_engaged argument to validate
           */
          static void validate(bool adapt_engaged) {
          }

          /**
           * Default value: true.
           *
           * @returns default
           */
          static bool default_value() {
            return true;
          }
        };

        /**
         * Number of iterations for eta adaptation.
         */
        struct adapt_iterations {
          /**
           * Description of adapt_iterations.
           *
           * @returns description
           */
          static std::string description() {
            return "Number of iterations for eta adaptation.";
          }

          /**
           * Validates adapt_iterations; must be greater than 0.
           *
           * @param[in] adapt_iterations argument to validate
           * @throw std::invalid_argument if adapt_iterations <= 0
           */
          static void validate(int adapt_iterations) {
            if (!(adapt_iterations > 0))
              throw std::invalid_argument("adapt_iterations must be greater "
                                          "than 0.");
          }

          /**
           * Default adapt_iterations.
           *
           * @returns default
           */
          static int default_value() {
            return 50;
          }
        };

        /**
         * Evaluate ELBO every Nth iteration
         */
        struct eval_elbo {
          /**
           * Description of eval_elbo. Evaluate ELBO at every
           * <code>eval_elbo</code> iterations.
           *
           * @returns description
           */
          static std::string description() {
            return "Number of interations between ELBO evaluations";
          }

          /**
           * Validates eval_elbo; must be greater than 0.
           *
           * @param[in] eval_elbo argument to validate
           * @throw std::invalid_argument if eval_elbo <= 0
           */
          static void validate(int eval_elbo) {
            if (!(eval_elbo > 0))
              throw std::invalid_argument("eval_elbo must be greater than 0.");
          }

          /**
           * Default eval_elbo; defaults to 100.
           *
           * @returns default
           */
          static int default_value() {
            return 100;
          }
        };

        /**
         * Number of approximate posterior output draws to save.
         */
        struct output_draws {
          /**
           * Description of output_draws.
           *
           * @returns description
           */
          static std::string description() {
            return "Number of approximate posterior output draws to save.";
          }

          /**
           * Validates outpu_draws; must be greater than or equal to 0.
           *
           * @param[in] output_draws argument to validate
           * @throw std::invalid_argument if output_draws < 0
           */
          static void validate(int output_draws) {
            if (!(output_draws >= 0))
              throw std::invalid_argument("output_draws must be greater than "
                                          "or equal to 0.");
          }

          /**
           * Default output_samples; defaults to 1000.
           *
           * @returns default
           */
          static int default_value() {
            return 1000;
          }
        };

      }
    }
  }
}
#endif
