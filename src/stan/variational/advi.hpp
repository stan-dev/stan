#ifndef STAN_VARIATIONAL_ADVI_HPP
#define STAN_VARIATIONAL_ADVI_HPP

#include <stan/math.hpp>
#include <stan/io/dump.hpp>
#include <stan/model/util.hpp>
#include <stan/services/io/write_iteration_csv.hpp>
#include <stan/services/io/write_iteration.hpp>
#include <stan/services/error_codes.hpp>
#include <stan/services/variational/print_progress.hpp>
#include <stan/variational/families/normal_fullrank.hpp>
#include <stan/variational/families/normal_meanfield.hpp>
#include <boost/circular_buffer.hpp>
#include <boost/lexical_cast.hpp>
#include <algorithm>
#include <limits>
#include <numeric>
#include <ostream>
#include <vector>
#include <queue>
#include <string>

namespace stan {

  namespace variational {

    /**
     * AUTOMATIC DIFFERENTIATION VARIATIONAL INFERENCE
     *
     * Runs "black box" variational inference by applying stochastic gradient
     * ascent in order to maximize the Evidence Lower Bound for a given model
     * and variational family.
     *
     * @tparam M                     class of model
     * @tparam Q                     class of variational distribution
     * @tparam BaseRNG               class of random number generator
     * @param  m                     stan model
     * @param  cont_params           initialization of continuous parameters
     * @param  rng                   random number generator
     * @param  n_monte_carlo_grad    number of samples for gradient computation
     * @param  n_monte_carlo_elbo    number of samples for ELBO computation
     * @param  eval_elbo             evaluate ELBO at every "eval_elbo" iters
     * @param  n_posterior_samples   number of samples to draw from posterior
     * @param  print_stream          stream for convergence assessment output
     * @param  output_stream         stream for parameters output
     * @param  diagnostic_stream     stream for ELBO output
     */
    template <class M, class Q, class BaseRNG>
    class advi {
    public:
      advi(M& m,
           Eigen::VectorXd& cont_params,
           BaseRNG& rng,
           int n_monte_carlo_grad,
           int n_monte_carlo_elbo,
           int eval_elbo,
           int n_posterior_samples,
           std::ostream* print_stream,
           std::ostream* output_stream,
           std::ostream* diagnostic_stream) :
        model_(m),
        cont_params_(cont_params),
        rng_(rng),
        n_monte_carlo_grad_(n_monte_carlo_grad),
        n_monte_carlo_elbo_(n_monte_carlo_elbo),
        eval_elbo_(eval_elbo),
        n_posterior_samples_(n_posterior_samples),
        print_stream_(print_stream),
        out_stream_(output_stream),
        diag_stream_(diagnostic_stream) {
        static const char* function = "stan::variational::advi";
        stan::math::check_positive(function,
                                 "Number of Monte Carlo samples for gradients",
                                 n_monte_carlo_grad_);
        stan::math::check_positive(function,
                                 "Evaluate ELBO at every eval_elbo iteration",
                                 eval_elbo_);
        stan::math::check_positive(function,
                                 "Number of posterior samples for output",
                                 n_posterior_samples_);
        stan::math::check_positive(function,
                                 "Number of posterior samples for output",
                                 n_posterior_samples_);
      }

      /**
       * Calculates the Evidence Lower BOund (ELBO) by sampling from
       * the variational distribution and then evaluating the log joint,
       * adjusted by the entropy term of the variational distribution.
       *
       * @param[in] variational variational approximation at which to evaluate
       * the ELBO.
       * @return the evidence lower bound.
       * @throw std::domain_error If, after n_monte_carlo_elbo_ number of draws
       * from the variational distribution all give non-finite log joint
       * evaluations. This means that the model is severly ill conditioned or
       * that the variational distribution has somehow collapsed.
       */
      double calc_ELBO(const Q& variational) const {
        static const char* function =
          "stan::variational::advi::calc_ELBO";

        double elbo = 0.0;
        int dim = variational.dimension();
        Eigen::VectorXd zeta(dim);

        int n_dropped_evaluations = 0;
        for (int i = 0; i < n_monte_carlo_elbo_;) {
          variational.sample(rng_, zeta);
          try {
            double log_prob = model_.template log_prob<false, true>(zeta,
              print_stream_);
            stan::math::check_finite(function, "log_prob", log_prob);
            elbo += log_prob;
            ++i;
          } catch (const std::domain_error& e) {
            ++n_dropped_evaluations;
            if (n_dropped_evaluations >= n_monte_carlo_elbo_) {
              const char* name = "The number of dropped evaluations";
              const char* msg1 = "has reached its maximum amount (";
              const char* msg2 = "). Your model may be either severely "
                "ill-conditioned or misspecified.";
              stan::math::domain_error(function, name, n_monte_carlo_elbo_,
                msg1, msg2);
            }
          }
        }
        elbo /= n_monte_carlo_elbo_;

        elbo += variational.entropy();

        return elbo;
      }

      /**
       * Calculates the "black box" gradient of the ELBO.
       *
       * @param[in] variational variational approximation at which to evaluate
       * the ELBO.
       * @param[out] elbo_grad gradient of ELBO with respect to variational
       * approximation.
       */
      void calc_ELBO_grad(const Q& variational, Q& elbo_grad) const {
        static const char* function =
          "stan::variational::advi::calc_ELBO_grad";

        stan::math::check_size_match(function,
                        "Dimension of elbo_grad", elbo_grad.dimension(),
                        "Dimension of variational q", variational.dimension());
        stan::math::check_size_match(function,
                        "Dimension of variational q", variational.dimension(),
                        "Dimension of variables in model", cont_params_.size());

        variational.calc_grad(elbo_grad,
                              model_, cont_params_, n_monte_carlo_grad_, rng_,
                              print_stream_);
      }

      /**
       * Heuristic grid search to adapt eta to the scale of the problem.
       *
       * @param[in] variational initial variational distribution.
       * @param adapt_iterations number of iterations to spend doing stochastic
       * gradient ascent at each proposed eta value.
       * @return adapted (tuned) value of eta via heuristic grid search
       * @throw std::domain_error If either (a) the initial ELBO cannot be
       * computed at the initial variational distribution, (b) all step-size
       * proposals in eta_sequence fail.
       */
      double adapt_eta(Q& variational, int adapt_iterations) const {
        static const char* function = "stan::variational::advi::adapt_eta";

        stan::math::check_positive(function,
                                   "Number of adaptation iterations",
                                   adapt_iterations);

        if (print_stream_) {
          *print_stream_ << "Begin eta adaptation." << std::endl;
        }

        // Sequence of eta values to try during adaptation
        const int eta_sequence_size = 5;
        double eta_sequence[eta_sequence_size] = {100, 10, 1, 0.1, 0.01};

        // Initialize ELBO tracking variables
        double elbo      = -std::numeric_limits<double>::max();
        double elbo_best = -std::numeric_limits<double>::max();
        double elbo_init;
        try {
            elbo_init = calc_ELBO(variational);
        } catch (const std::domain_error& e) {
            const char* name = "Cannot compute ELBO using the initial "
                  "variational distribution.";
            const char* msg1 = "Your model may be either "
                  "severely ill-conditioned or misspecified.";
            stan::math::domain_error(function, name, "", msg1);
        }

        // Variational family to store gradients
        Q elbo_grad = Q(model_.num_params_r());

        // Adaptive step-size sequence
        Q history_grad_squared = Q(model_.num_params_r());
        double tau = 1.0;
        double pre_factor  = 0.9;
        double post_factor = 0.1;

        double eta_best = 0.0;
        double eta;
        double eta_scaled;

        bool do_more_tuning = true;
        int eta_sequence_index = 0;
        while (do_more_tuning) {
          // Try next eta
          eta = eta_sequence[eta_sequence_index];

          int print_progress_m;
          for (int iter_tune = 1; iter_tune <= adapt_iterations; ++iter_tune) {
            print_progress_m = eta_sequence_index
                               * adapt_iterations + iter_tune;
            stan::services::variational::print_progress(
              print_progress_m, 0, adapt_iterations*eta_sequence_size,
              adapt_iterations, true, "", "", *print_stream_);

            // (ROBUST) Compute gradient of ELBO. It's OK if it diverges.
            // We'll try a smaller eta.
            try {
              calc_ELBO_grad(variational, elbo_grad);
            } catch (const std::domain_error& e) {
              elbo_grad.set_to_zero();
            }

            // Update step-size
            if (iter_tune == 1) {
              history_grad_squared += elbo_grad.square();
            } else {
              history_grad_squared = pre_factor * history_grad_squared
                                     + post_factor * elbo_grad.square();
            }
            eta_scaled = eta / sqrt(static_cast<double>(iter_tune));

            // Stochastic gradient update
            variational += eta_scaled * elbo_grad
                          / (tau + history_grad_squared.sqrt());
          }

          // (ROBUST) Compute ELBO. It's OK if it has diverged.
          try {
            elbo = calc_ELBO(variational);
          } catch (const std::domain_error& e) {
            elbo = -std::numeric_limits<double>::max();
          }

          // Check if:
          // (1) ELBO at current eta is worse than the best ELBO
          // (2) the best ELBO hasn't gotten worse than the initial ELBO
          if (elbo < elbo_best && elbo_best > elbo_init) {
            if (print_stream_) {
              *print_stream_ << "Success!"
                << " Found best value [eta = " << eta_best
                << "]";
              if (eta_sequence_index < eta_sequence_size - 1)
                *print_stream_ << " earlier than expected."
                  << std::endl << std::endl;
              else
                *print_stream_ << "." << std::endl << std::endl;
            }
            do_more_tuning = false;
          } else {
            if (eta_sequence_index < eta_sequence_size - 1) {
              // Reset
              elbo_best = elbo;
              eta_best = eta;
            } else {
              // No more eta values to try, so use current eta if it
              // didn't diverge or fail if it did diverge
              if (elbo > elbo_init) {
                if (print_stream_)
                  *print_stream_ << "Success!"
                    << " Found best value [eta = " << eta_best
                    << "]." << std::endl << std::endl;
                eta_best = eta;
                do_more_tuning = false;
              } else {
                const char* name = "All proposed step-sizes";
                const char* msg1 = "failed. Your model may be either "
                  "severely ill-conditioned or misspecified.";
                stan::math::domain_error(function, name, "", msg1);
              }
            }
            // Reset
            history_grad_squared.set_to_zero();
          }
          ++eta_sequence_index;
          variational = Q(cont_params_);
        }
        return eta_best;
      }

      /**
       * Runs stochastic gradient ascent with an adaptive stepsize sequence.
       *
       * @param[in,out] variational initia variational distribution
       * @param eta stepsize scaling parameter
       * @param tol_rel_obj relative tolerance parameter for convergence
       * @param max_iterations max number of iterations to run algorithm
       * @throw std::domain_error If the ELBO or its gradient is ever
       * non-finite, at any iteration
       */
      void stochastic_gradient_ascent(Q& variational,
                                 double eta,
                                 double tol_rel_obj,
                                 int max_iterations) const {
        static const char* function =
          "stan::variational::advi.stochastic_gradient_ascent";

        stan::math::check_positive(function, "Eta stepsize", eta);
        stan::math::check_positive(function,
                                   "Relative objective function tolerance",
                                   tol_rel_obj);
        stan::math::check_positive(function,
                                   "Maximum iterations",
                                   max_iterations);

        // Gradient parameters
        Q elbo_grad = Q(model_.num_params_r());

        // Stepsize sequence parameters
        Q history_grad_squared = Q(model_.num_params_r());
        double tau = 1.0;
        double pre_factor  = 0.9;
        double post_factor = 0.1;
        double eta_scaled;

        // Initialize ELBO and convergence tracking variables
        double elbo(0.0);
        double elbo_best      = -std::numeric_limits<double>::max();
        double elbo_prev      = -std::numeric_limits<double>::max();
        double delta_elbo     = std::numeric_limits<double>::max();
        double delta_elbo_ave = std::numeric_limits<double>::max();
        double delta_elbo_med = std::numeric_limits<double>::max();

        // Heuristic to estimate how far to look back in rolling window
        int cb_size = static_cast<int>(
                std::max(0.1*max_iterations/static_cast<double>(eval_elbo_),
                         2.0));
        boost::circular_buffer<double> elbo_diff(cb_size);

        if (print_stream_) {
          *print_stream_ << "Begin stochastic gradient ascent." << std::endl
                         << "  iter"
                         << "       ELBO"
                         << "   delta_ELBO_mean"
                         << "   delta_ELBO_med"
                         << "   notes "
                         << std::endl;
        }

        // Timing variables
        clock_t start = clock();
        clock_t end;
        double delta_t;

        // Main loop
        std::vector<double> print_vector;
        bool do_more_iterations = true;
        for (int iter_counter = 1; do_more_iterations; ++iter_counter) {
          // Compute gradient using Monte Carlo integration
          calc_ELBO_grad(variational, elbo_grad);

          // Update step-size
          if (iter_counter == 1) {
            history_grad_squared += elbo_grad.square();
          } else {
            history_grad_squared = pre_factor * history_grad_squared
                                   + post_factor * elbo_grad.square();
          }
          eta_scaled = eta / sqrt(static_cast<double>(iter_counter));

          // Stochastic gradient update
          variational += eta_scaled * elbo_grad
                         / (tau + history_grad_squared.sqrt());

          // Check for convergence every "eval_elbo_"th iteration
          if (iter_counter % eval_elbo_ == 0) {
            elbo_prev = elbo;
            elbo = calc_ELBO(variational);
            if (elbo > elbo_best)
              elbo_best = elbo;
            delta_elbo = rel_difference(elbo, elbo_prev);
            elbo_diff.push_back(delta_elbo);
            delta_elbo_ave = std::accumulate(elbo_diff.begin(),
                              elbo_diff.end(), 0.0)
                              / static_cast<double>(elbo_diff.size());
            delta_elbo_med = circ_buff_median(elbo_diff);
            if (print_stream_) {
              *print_stream_
                        << "  "
                        << std::setw(4) << iter_counter
                        << "  "
                        << std::right << std::setw(9) << std::setprecision(1)
                        << elbo
                        << "  "
                        << std::setw(16) << std::fixed << std::setprecision(3)
                        << delta_elbo_ave
                        << "  "
                        << std::setw(15) << std::fixed << std::setprecision(3)
                        << delta_elbo_med;
            }

            if (diag_stream_) {
              end = clock();
              delta_t = static_cast<double>(end - start) / CLOCKS_PER_SEC;

              print_vector.clear();
              print_vector.push_back(delta_t);
              print_vector.push_back(elbo);
              services::io::write_iteration_csv(*diag_stream_,
                                                iter_counter, print_vector);
            }

            if (delta_elbo_ave < tol_rel_obj) {
              if (print_stream_)
                *print_stream_ << "   MEAN ELBO CONVERGED";
              do_more_iterations = false;
            }

            if (delta_elbo_med < tol_rel_obj) {
              if (print_stream_)
                *print_stream_ << "   MEDIAN ELBO CONVERGED";
              do_more_iterations = false;
            }

            if (iter_counter > 10 * eval_elbo_) {
              if (delta_elbo_med > 0.5 || delta_elbo_ave > 0.5) {
                if (print_stream_)
                  *print_stream_ << "   MAY BE DIVERGING... INSPECT ELBO";
              }
            }

            if (print_stream_)
              *print_stream_ << std::endl;

            if (do_more_iterations == false &&
                rel_difference(elbo, elbo_best) > 0.05) {
              if (print_stream_)
                *print_stream_
                  << "Informational Message: The ELBO at a previous "
                  << "iteration is larger than the ELBO upon "
                  << "convergence!"
                  << std::endl
                  << "This variational approximation may not "
                  << "have converged to a good optimum."
                  << std::endl;
            }
          }

          if (iter_counter == max_iterations) {
            if (print_stream_)
              *print_stream_
                << "Informational Message: The maximum number of "
                << "iterations is reached! The algorithm may not have "
                << "converged."
                << std::endl
                << "This variational approximation is not "
                << "guaranteed to be meaningful."
                << std::endl;
            do_more_iterations = false;
          }
        }
      }

      /**
       * Runs ADVI and writes to output.
       *
       * @param  eta              eta parameter of stepsize sequence
       * @param  adapt_engaged    boolean flag for eta adaptation
       * @param  adapt_iterations number of iterations for eta adaptation
       * @param  tol_rel_obj      relative tolerance parameter for convergence
       * @param  max_iterations   max number of iterations to run algorithm
       */
      int run(double eta, bool adapt_engaged, int adapt_iterations,
              double tol_rel_obj, int max_iterations) const {
        if (diag_stream_) {
          *diag_stream_ << "iter,time_in_seconds,ELBO" << std::endl;
        }

        // Initialize variational approximation
        Q variational = Q(cont_params_);

        if (adapt_engaged) {
          eta = adapt_eta(variational, adapt_iterations);
          if (out_stream_) {
            *out_stream_ << "# Stepsize adaptation complete." << std::endl
                         << "# eta = " << eta << std::endl;
          }
        }

        stochastic_gradient_ascent(variational, eta,
                                   tol_rel_obj, max_iterations);

        // Write mean of posterior approximation on first output line
        cont_params_ = variational.mean();
        std::vector<double> cont_vector(cont_params_.size());
        for (int i = 0; i < cont_params_.size(); ++i)
          cont_vector.at(i) = cont_params_(i);
        std::vector<int> disc_vector;

        if (out_stream_) {
          services::io::write_iteration(*out_stream_, model_, rng_,
                                        0, cont_vector, disc_vector,
                                        print_stream_);
        }

        // Draw more samples from posterior and write on subsequent lines
        if (out_stream_) {
          if (print_stream_) {
            *print_stream_ << std::endl
                           << "Drawing "
                           << n_posterior_samples_
                           << " samples from the approximate posterior... ";
            print_stream_->flush();
          }

          for (int n = 0; n < n_posterior_samples_; ++n) {
            variational.sample(rng_, cont_params_);
            for (int i = 0; i < cont_params_.size(); ++i) {
              cont_vector.at(i) = cont_params_(i);
            }
            services::io::write_iteration(*out_stream_, model_, rng_,
                          0, cont_vector, disc_vector, print_stream_);
          }

          if (print_stream_) {
            *print_stream_ << "COMPLETED." << std::endl;
          }
        }

        return stan::services::error_codes::OK;
      }

      // TODO(akucukelbir): move these things to stan math and test there

      /**
       * Compute the median of a circular buffer.
       *
       * @param  cb circular buffer with some number of values in it.
       * @return    median of values in circular buffer.
       */
      double circ_buff_median(const boost::circular_buffer<double>& cb) const {
          // FIXME: naive implementation; creates a copy as a vector
          std::vector<double> v;
          for (boost::circular_buffer<double>::const_iterator i = cb.begin();
                i != cb.end(); ++i) {
            v.push_back(*i);
          }

          size_t n = v.size() / 2;
          std::nth_element(v.begin(), v.begin()+n, v.end());
          return v[n];
      }

      /**
       * Compute the relative difference between two double values.
       *
       * @param  prev previous value
       * @param  curr current value
       * @return      absolutely value of relative difference
       */
      double rel_difference(double prev, double curr) const {
        return std::fabs((curr - prev) / prev);
      }

    protected:
      M& model_;
      Eigen::VectorXd& cont_params_;
      BaseRNG& rng_;
      int n_monte_carlo_grad_;
      int n_monte_carlo_elbo_;
      int eval_elbo_;
      int n_posterior_samples_;
      std::ostream* print_stream_;
      std::ostream* out_stream_;
      std::ostream* diag_stream_;
    };
  }  // variational
}  // stan

#endif
