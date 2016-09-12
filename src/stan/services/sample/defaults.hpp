#ifndef STAN_SERVICES_SAMPLE_DEFAULTS_HPP
#define STAN_SERVICES_SAMPLE_DEFAULTS_HPP

#include <stdexcept>
#include <string>

namespace stan {
  namespace services {
    namespace sample {

      /**
       * Number of sampling iterations.
       */
      struct num_samples {

        /**
         * Description of num_samples.
         *
         * @returns description
         */
        static std::string description() {
          return "Number of sampling iterations.";
        }

        /**
         * Validates num_samples; num_samples must be greater than or equal to 0.
         *
         * @param[in] num_samples argument to validate
         * @throw std::invalid argument if num_samples < 0
         */
        static void validate(int num_samples) {
          if (!(num_samples >= 0))
            throw std::invalid_argument("num_samples must be greater than or equal to 0.");
        }

        /**
         * Default num_samples: 1000.
         *
         * @returns default
         */
        static int default_value() {
          return 1000;
        }
      };
      
      /**
       * Number of warmup iterations.
       */
      struct num_warmup {

        /**
         * Description of num_warmup.
         *
         * @returns description
         */
        static std::string description() {
          return "Number of warmup iterations.";
        }

        /**
         * Validates num_warmup; num_warmup must be greater than or
         * equal to 0.
         *
         * @param[in] num_warmup argument to validate
         * @throw std::invalid argument if num_warmup < 0
         */
        static void validate(int num_warmup) {
          if (!(num_warmup >= 0))
            throw std::invalid_argument("num_warmup must be greater "
                                        "than or equal to 0.");
        }

        /**
         * Default num_warmup: 1000.
         *
         * @returns default
         */
        static int default_value() {
          return 1000;
        }
      };

      /**
       * Save warmup iterations to output.
       */
      struct save_warmup {

        /**
         * Description of save_warmup.
         *
         * @returns description
         */
        static std::string description() {
          return "Save warmup iterations to output.";
        }

        /**
         * Validates save_warmup. This is a noop.
         *
         * @param[in] save_warmup argument to validate
         */
        static void validate(bool save_warmup) {
        }

        /**
         * Default save_warmup: false.
         *
         * @returns default
         */
        static bool default_value() {
          return false;
        }
      };
      
      /**
       * Period between saved samples.
       */
      struct thin {

        /**
         * Description of thin.
         *
         * @returns description
         */
        static std::string description() {
          return "Period between saved samples.";
        }

        /**
         * Validates thin; thin must be greater than 0.
         *
         * @param[in] thin argument to validate
         * @throw std::invalid argument if thin <= 0
         */
        static void validate(int thin) {
          if (!(thin > 0))
            throw std::invalid_argument("thin must be greater than 0.");
        }

        /**
         * Default thin: 1.
         *
         * @returns default
         */
        static int default_value() {
          return 1;
        }
      };
      
      /**
       * Indicates whether adaptation is engaged.
       */
      struct adaptation_engaged {

        /**
         * Description of adaptation_engaged.
         *
         * @returns description
         */
        static std::string description() {
          return "Indicates whether adaptation is engaged.";
        }

        /**
         * Validates adaptation_engaged. This is a noop.
         *
         * @param[in] adaptation_engaged argument to validate
         */
        static void validate(bool adaptation_engaged) {
        }

        /**
         * Default adaptation_engaged: true.
         *
         * @returns default
         */
        static bool default_value() {
          return true;
        }
      };


      /**
       * Adaptation regularization scale.
       */
      struct gamma {

        /**
         * Description of gamma.
         *
         * @returns description
         */
        static std::string description() {
          return "Adaptation regularization scale.";
        }

        /**
         * Validates gamma; gamma must be greater than 0.
         *
         * @param[in] gamma argument to validate
         * @throw std::invalid argument if gamma <= 0
         */
        static void validate(double gamma) {
          if (!(gamma > 0))
            throw std::invalid_argument("gamma must be greater than 0.");
        }

        /**
         * Default gamma: 0.05.
         *
         * @returns default
         */
        static double default_value() {
          return 0.05;
        }
      };

      /**
       * Adaptation relaxation exponent.
       */
      struct kappa {

        /**
         * Description of kappa.
         *
         * @returns description
         */
        static std::string description() {
          return "Adaptation relaxation exponent.";
        }

        /**
         * Validates kappa; kappa must be greater than 0.
         *
         * @param[in] kappa argument to validate
         * @throw std::invalid argument if kappa <= 0
         */
        static void validate(double kappa) {
          if (!(kappa > 0))
            throw std::invalid_argument("kappa must be greater than 0.");
        }

        /**
         * Default kappa: 0.75.
         *
         * @returns default
         */
        static double default_value() {
          return 0.75;
        }
      };
      
      /**
       * Adaptation iteration offset.
       */
      struct t0 {

        /**
         * Description of t0.
         *
         * @returns description
         */
        static std::string description() {
          return "Adaptation iteration offset.";
        }

        /**
         * Validates t0; t0 must be greater than 0.
         *
         * @param[in] t0 argument to validate
         * @throw std::invalid argument if t0 <= 0
         */
        static void validate(double t0) {
          if (!(t0 > 0))
            throw std::invalid_argument("t0 must be greater than 0.");
        }

        /**
         * Default t0: 10.
         *
         * @returns default
         */
        static double default_value() {
          return 10;
        }
      };

      /**
       * Width of initial fast adaptation interval.
       */
      struct init_buffer {

        /**
         * Description of init_buffer.
         *
         * @returns description
         */
        static std::string description() {
          return "Width of initial fast adaptation interval.";
        }

        /**
         * Validates init_buffer. This is a noop.
         *
         * @param[in] init_buffer argument to validate
         */
        static void validate(unsigned int init_buffer) {
        }

        /**
         * Default init_buffer: 75.
         *
         * @returns default
         */
        static unsigned int default_value() {
          return 75;
        }
      };

      /**
       * Width of final fast adaptation interval.
       */
      struct term_buffer {

        /**
         * Description of term_buffer.
         *
         * @returns description
         */
        static std::string description() {
          return "Width of final fast adaptation interval.";
        }

        /**
         * Validates term_buffer. This is a noop.
         *
         * @param[in] term_buffer argument to validate
         */
        static void validate(unsigned int term_buffer) {
        }

        /**
         * Default term_buffer: 50
         *
         * @returns default
         */
        static unsigned int default_value() {
          return 50;
        }
      };

      /**
       * Initial width of slow adaptation interval.
       */
      struct window {

        /**
         * Description of window.
         *
         * @returns description
         */
        static std::string description() {
          return "Initial width of slow adaptation interval.";
        }

        /**
         * Validates window. This is a noop.
         *
         * @param[in] window argument to validate
         */
        static void validate(unsigned int window) {
        }

        /**
         * Default window: 25.
         *
         * @returns default
         */
        static unsigned int default_value() {
          return 25;
        }
      };

      /**
       * Total integration time for Hamiltonian evolution.
       */
      struct int_time {

        /**
         * Description of int_time.
         *
         * @returns description
         */
        static std::string description() {
          return "Total integration time for Hamiltonian evolution.";
        }

        /**
         * Validates int_time. int_time must be greater than 0.
         *
         * @param[in] int_time argument to validate
         * @throw std::invalid argument if int_time <= 0
         */
        static void validate(double int_time) {
          if (!(int_time > 0))
            throw std::invalid_argument("int_time must be greater than 0.");
        }

        /**
         * Default int_time: 2 * pi.
         *
         * @returns default
         */
        static double default_value() {
          return 6.28318530717959;
        }
      };

      /**
       * Maximum tree depth.
       */
      struct max_depth {

        /**
         * Description of max_depth.
         *
         * @returns description
         */
        static std::string description() {
          return "Maximum tree depth.";
        }

        /**
         * Validates max_depth; max_depth must be greater than 0.
         *
         * @param[in] max_depth argument to validate
         * @throw std::invalid argument if max_depth <= 0
         */
        static void validate(int max_depth) {
          if (!(max_depth > 0))
            throw std::invalid_argument("max_depth must be greater than 0.");
        }

        /**
         * Default max_depth: 10.
         *
         * @returns default
         */
        static int default_value() {
          return 10;
        }
      };

      /**
       * Step size for discrete evolution
       */
      struct stepsize {

        /**
         * Description of stepsize.
         *
         * @returns description
         */
        static std::string description() {
          return "Step size for discrete evolution.";
        }

        /**
         * Validates stepsize; stepsize must be greater than 0.
         *
         * @param[in] stepsize argument to validate
         * @throw std::invalid argument if stepsize <= 0
         */
        static void validate(double stepsize) {
          if (!(stepsize > 0))
            throw std::invalid_argument("stepsize must be greater than 0.");
        }

        /**
         * Default stepsize: 1.
         *
         * @returns default
         */
        static double default_value() {
          return 1;
        }
      };

      /**
       * Uniformly random jitter of the stepsize, in percent.
       */
      struct stepsize_jitter {

        /**
         * Description of stepsize_jitter.
         *
         * @returns description
         */
        static std::string description() {
          return "Uniformly random jitter of the stepsize, in percent.";
        }

        /**
         * Validates stepsize_jitter; stepsize_jitter must be between 0 and 1.
         *
         * @param[in] stepsize_jitter argument to validate
         * @throw std::invalid argument if stepsize_jitter < 0 or stepsize_jitter > 1
         */
        static void validate(double stepsize_jitter) {
          if (!(stepsize_jitter >= 0 && stepsize_jitter <= 1))
            throw std::invalid_argument("stepsize_jitter must be between 0 and 1.");
        }

        /**
         * Default stepsize_jitter: 0.
         *
         * @returns default
         */
        static double default_value() {
          return 0;
        }
      };


      
    }
  }
}
#endif
