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
            return "number of Monte Carlo draws for gradient computation";
          }

          /**
           * Validates gradient_samples; must be greater than 0.
           *
           * @throw std::invalid_argument if invalid
           */
          static void validate(int gradient_samples) {
            if (!gradient_samples > 0)
              throw std::invalid_argument("gradient_samples must be greater than 0");
          }

          /**
           * Default number of grad samples: 1
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
            return "Number of Monte Carlo samples for estimate of ELBO";
          }

          /**
           * Validates elbo_samples; must be greater than 0.
           *
           * @throw std::invalid_argument if not valid
           */
          static void validate(double elbo_samples) {
            if (!elbo_sample > 0)
              throw std::invalid_argument("elbo_samples must be greater than 0"):
          }

          /**
           * Default number of elbo samples: 1100
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
            return "maximum number of ADVI iterations";
          }

          /**
           * Validates max_iterations; max_iterations must be greater than 0.
           *
           * @throw std::invalid_argument if invalid
           */
          static void validate(int max_iterations) {
            if (!max_iterations > 0)
              throw std::invalid_argument("max_iterations must be greater than 0");
          }

          /**
           * Default
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
           * Description of tol_rel_obj. This is the relative
           * tolerance parameter for convergence.x
           *
           * @returns description
           */
          static std::string description() {
            return "relative tolerance parameter for convergence";
          }

          /**
           * Validates tol_rel_obj; must be greater than 0.
           *
           * @throw std::invalid_argument if invalid
           */
          static void validate(double tol_rel_obj) {
            if (!tol_rel_obj > 0)
              throw std::invalid_argument("");
          }

          /**
           * Default relative tolerance parameter for convergence
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
           * Description of eta. Stepsize scaling parameter for variational inference
           *
           * @returns description
           */
          static std::string description() {
            return "Stepsize scaling parameter for variational inference";
          }

          /**
           * Validates eta; must be greater than 0.
           *
           * @throw std::invalid_argument if invalid
           */
          static void validate(double eta) {
            if (eta > 0)
              throw std::invalid_argument("eta must be greater than 0");
          }

          /**
           * Default
           *
           * @returns default
           */
          static double default_value() {
            return 1.0;
          }
        };

        /**
         * flag for eta adaptation
         */
        struct adapt_engaged {

          /**
           * Description of adapt_engaged. Boolean flag for eta adaptation
           *
           * @returns description
           */
          static std::string description() {
            return "Boolean flag for eta adaptation";
          }

          /**
           * Validates adapt_engaged. This is a noop
           *
           * @throw std::invalid_argument if invalid
           */
          static void validate(bool adapt_engaged) {
          }

          /**
           * Default
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
           * Description of adapt_iterations. Number of iterations for eta adaptation
           *
           * @returns description
           */
          static std::string description() {
            return "Number of iterations for eta adaptation";
          }

          /**
           * Validates adapt_iterations; must be greater than 0.
           *
           * @throw std::invalid_argument if invalid
           */
          static void validate(int adapt_iterations) {
            if (!adapt_iterations > 0)
              throw std::invalid_argument("");
          }

          /**
           * Default
           *
           * @returns default
           */
          static int default_value() {
            return 50;
          }
        };




        // int eval_elbo, = evaluate ELBO at every "eval_elbo" iters
        // int output_samples, = n_posterior_samples number of samples to draw from posterior


      }
    }
  }
}
#endif
