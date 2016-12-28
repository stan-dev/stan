#ifndef STAN_SERVICES_OPTIMIZE_DEFAULTS_HPP
#define STAN_SERVICES_OPTIMIZE_DEFAULTS_HPP

#include <stdexcept>
#include <string>

namespace stan {
  namespace services {
    namespace optimize {

      /**
       * Line search step size for first iteration.
       */
      struct init_alpha {
        /**
         * Description of init_alpha.
         *
         * @return description
         */
        static std::string description() {
          return "Line search step size for first iteration.";
        }

        /**
         * Validates init_alpha; init_alpha must be greater than 0.
         *
         * @param[in] init_alpha argument to validate
         * @throw std::invalid argument if init_alpha <= 0
         */
        static void validate(double init_alpha) {
          if (!(init_alpha > 0))
            throw std::invalid_argument("init_alpha must be greater than 0.");
        }

        /**
         * Default init_alpha: 0.001.
         *
         * @return default
         */
        static double default_value() {
          return 0.001;
        }
      };

      /**
       * Convergence tolerance on absolute changes in objective function value.
       */
      struct tol_obj {
        /**
         * Description of tol_obj.
         *
         * @return description
         */
        static std::string description() {
          return "Convergence tolerance on absolute changes in objective "
            "function value.";
        }

        /**
         * Validates tol_obj; tol_obj must be greater than or equal to 0.
         *
         * @param[in] tol_obj argument to validate
         * @throw std::invalid argument if tol_obj < 0
         */
        static void validate(double tol_obj) {
          if (!(tol_obj >= 0))
            throw std::invalid_argument("tol_obj must be greater "
                                        "than or equal to 0.");
        }

        /**
         * Default tol_obj: 1e-12.
         *
         * @return default
         */
        static double default_value() {
          return 1e-12;
        }
      };

      /**
       * Convergence tolerance on relative changes in objective function value.
       */
      struct tol_rel_obj {
        /**
         * Description of tol_rel_obj.
         *
         * @return description
         */
        static std::string description() {
          return "Convergence tolerance on relative changes in "
            "objective function value.";
        }

        /**
         * Validates tol_rel_obj; tol_rel_obj must be greater than or equal
         * to 0.
         *
         * @param[in] tol_rel_obj argument to validate
         * @throw std::invalid argument if tol_rel_obj < 0
         */
        static void validate(double tol_rel_obj) {
          if (!(tol_rel_obj >= 0))
            throw std::invalid_argument("tol_rel_obj must be greater "
                                        "than or equal to 0");
        }

        /**
         * Default tol_rel_obj: 10000.
         *
         * @return default
         */
        static double default_value() {
          return 10000;
        }
      };

      /**
       * Convergence tolerance on the norm of the gradient.
       */
      struct tol_grad {
        /**
         * Description of tol_grad.
         *
         * @return description
         */
        static std::string description() {
          return "Convergence tolerance on the norm of the gradient.";
        }

        /**
         * Validates tol_grad; tol_grad must be greater than or equal to 0.
         *
         * @param[in] tol_grad argument to validate
         * @throw std::invalid argument if tol_grad < 0
         */
        static void validate(double tol_grad) {
          if (!(tol_grad >= 0))
            throw std::invalid_argument("tol_grad must be greater "
                                        "than or equal to 0");
        }

        /**
         * Default tol_grad: 1e-8.
         *
         * @return default
         */
        static double default_value() {
          return 1e-8;
        }
      };


      /**
       * Convergence tolerance on the relative norm of the gradient.
       */
      struct tol_rel_grad {
        /**
         * Description of tol_rel_grad.
         *
         * @return description
         */
        static std::string description() {
          return "Convergence tolerance on the relative norm of the gradient.";
        }

        /**
         * Validates tol_rel_grad; tol_rel_grad must be greater than
         * or equal to 0.
         *
         * @param[in] tol_rel_grad argument to validate
         * @throw std::invalid argument if tol_rel_grad < 0
         */
        static void validate(double tol_rel_grad) {
          if (!(tol_rel_grad >= 0))
            throw std::invalid_argument("tol_rel_grad must be greater "
                                        "than or equal to 0.");
        }

        /**
         * Default tol_rel_grad: 10000000
         *
         * @return default
         */
        static double default_value() {
          return 10000000;
        }
      };

      /**
       * Convergence tolerance on changes in parameter value.
       */
      struct tol_param {
        /**
         * Description of tol_param.
         *
         * @return description
         */
        static std::string description() {
          return "Convergence tolerance on changes in parameter value.";
        }

        /**
         * Validates tol_param; tol_param must be greater than or equal to 0.
         *
         * @param[in] tol_param argument to validate
         * @throw std::invalid argument if tol_param < 0
         */
        static void validate(double tol_param) {
          if (!(tol_param >= 0))
            throw std::invalid_argument("tol_param");
        }

        /**
         * Default tol_param: 1e-08.
         *
         * @return default
         */
        static double default_value() {
          return 1e-08;
        }
      };

      /**
       * Amount of history to keep for L-BFGS.
       */
      struct history_size {
        /**
         * Description of history_size.
         *
         * @return description
         */
        static std::string description() {
          return "Amount of history to keep for L-BFGS.";
        }

        /**
         * Validates history_size; history_size must be greater than 0.
         *
         * @param[in] history_size argument to validate
         * @throw std::invalid argument if history_size <= 0
         */
        static void validate(int history_size) {
          if (!(history_size > 0))
            throw std::invalid_argument("history_size must be greater than 0.");
        }

        /**
         * Default history_size: 5.
         *
         * @return default
         */
        static int default_value() {
          return 5;
        }
      };

      /**
       * Total number of iterations.
       */
      struct iter {
        /**
         * Description of iter.
         *
         * @return description
         */
        static std::string description() {
          return "Total number of iterations.";
        }

        /**
         * Validates iter; iter must be greater than 0.
         *
         * @param[in] iter argument to validate
         * @throw std::invalid argument if iter <= 0
         */
        static void validate(int iter) {
          if (!(iter > 0))
            throw std::invalid_argument("iter must be greater than 0.");
        }

        /**
         * Default iter: 2000
         *
         * @return default
         */
        static int default_value() {
          return 2000;
        }
      };

      /**
       * Save optimization interations to output.
       */
      struct save_iterations {
        /**
         * Description of save_iterations.
         *
         * @return description
         */
        static std::string description() {
          return "Save optimization interations to output.";
        }

        /**
         * Validates save_iterations. This is a noop.
         *
         * @param[in] save_iterations argument to validate
         */
        static void validate(bool save_iterations) {
        }

        /**
         * Default save_iterations: false
         *
         * @return default
         */
        static bool default_value() {
          return false;
        }
      };

    }
  }
}
#endif
