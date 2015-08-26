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
#include <algorithm>
#include <limits>
#include <numeric>
#include <ostream>
#include <vector>
#include <queue>
#include <string>
#include <boost/lexical_cast.hpp>

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
     * @param  n_monte_carlo_grad    number of samples for gradient computation
     * @param  n_monte_carlo_elbo    number of samples for ELBO computation
     * @param  rng                   random number generator
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
           int n_monte_carlo_grad,
           int n_monte_carlo_elbo,
           BaseRNG& rng,
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
                                 "Number of Monte Carlo samples for ELBO",
                                 n_monte_carlo_elbo_);
        stan::math::check_positive(function,
                                 "Number of posterior samples for output",
                                 n_posterior_samples_);
        }

      /**
       * Calculates the Evidence Lower BOund (ELBO) by sampling from
       * the variational distribution and then evaluating the log joint,
       * adjusted by the entropy term of the variational distribution.
       *
       * @param variational variational distribution
       * @return evidence lower bound
       */
      double calc_ELBO(const Q& variational) const {
        static const char* function =
          "stan::variational::advi::calc_ELBO";

        double elbo = 0.0;
        int dim = variational.dimension();
        Eigen::VectorXd zeta(dim);

        int i = 0;
        int n_monte_carlo_drop = 0;
        while (i < n_monte_carlo_elbo_) {
          variational.sample(rng_, zeta);
          try {
            double energy_i = model_.template log_prob<false, true>(zeta,
              print_stream_);
            stan::math::check_finite(function, "energy_i", energy_i);
            elbo += energy_i;
            i += 1;
          } catch (std::exception& e) {
            this->write_error_msg_(print_stream_, e);
            n_monte_carlo_drop += 1;
            if (n_monte_carlo_drop >= n_monte_carlo_elbo_) {
              const char* name = "The number of dropped evaluations";
              const char* msg1 = "has reached its maximum amount (";
              int y = n_monte_carlo_elbo_;
              const char* msg2 = "). Your model may be either severely "
                "ill-conditioned or misspecified.";
              stan::math::domain_error(function, name, y, msg1, msg2);
            }
          }
        }

        // Divide to get Monte Carlo integral estimate
        elbo /= n_monte_carlo_elbo_;
        elbo += variational.entropy();

        return elbo;
      }

      /**
       * Calculates the "black box" gradient of the ELBO.
       *
       * @param variational variational distribution
       * @param elbo_grad   gradient of ELBO with respect to variational parameters
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
       * Adaptively set hyperparameters for ADVI.
       *
       * @param variational variational distribution
       * @param tuning_iter number of tuning iterations
       * @return optimal value of eta via grid search
       */
      double tune(Q& variational, int tuning_iter) const {
        static const char* function =
          "stan::variational::advi::tune";

        stan::math::check_positive(function,
                                   "Number of tuning iterations",
                                   tuning_iter);

        // Gradient parameters
        Q elbo_grad = Q(model_.num_params_r());

        // Learning rate parameters
        Q params_prop = Q(model_.num_params_r());
        double tau = 1.0;
        double pre_factor  = 0.9;
        double post_factor = 0.1;
        double eta_scaled;

        // Sequence of eta values to try during tuning
        std::queue<double> eta_sequence;
        eta_sequence.push(1.00);
        eta_sequence.push(0.50);
        eta_sequence.push(0.10);
        eta_sequence.push(0.05);
        eta_sequence.push(0.01);

        // Print progress
        int eta_sequence_size = eta_sequence.size();
        int m;
        if (print_stream_) {
          *print_stream_ << "Begin hyperparameter tuning." << std::endl;
        }

        // Initialize ELBO and convergence tracking variables
        double elbo(0.0);
        double elbo_best = -std::numeric_limits<double>::max();
        double elbo_init = calc_ELBO(variational);
        double eta_best(0.0);

        int iter_tune;
        bool do_more_tuning = true;
        while (do_more_tuning) {
          // Try next eta
          double eta = eta_sequence.front();
          eta_sequence.pop();

          for (iter_tune = 1; iter_tune <= tuning_iter; ++iter_tune) {
            m = (eta_sequence_size - eta_sequence.size() - 1)
              * tuning_iter + iter_tune; // # of total tuning iterations
            stan::services::variational::print_progress(
              m, 0, tuning_iter*eta_sequence_size,
              tuning_iter, true, "", "", *print_stream_);

            // Compute gradient of ELBO
            calc_ELBO_grad(variational, elbo_grad);

            // Update learning rate parameters
            if (iter_tune == 1) {
              params_prop += elbo_grad.square();
            } else {
              params_prop = pre_factor * params_prop
                            + post_factor * elbo_grad.square();
            }
            eta_scaled = eta / sqrt(static_cast<double>(iter_tune));

            // Stochastic gradient update
            variational += eta_scaled * elbo_grad / (tau + params_prop.sqrt());
          }
          elbo = calc_ELBO(variational);

          // Check if:
          // (1) ELBO at current eta is worse than the best ELBO
          // (2) the best ELBO hasn't actually diverged
          if (elbo < elbo_best && elbo_best > elbo_init) {
            if (print_stream_) {
              *print_stream_ << "Success!"
                << " Found best tuned hyperparameter"
                << " [eta = " << eta_best
                << "]";
              if (eta_sequence.size() > 0) {
                *print_stream_
                  << " earlier than expected.";
              }
              *print_stream_ << std::endl << std::endl;
            }
            do_more_tuning = false;
          } else {
            if (eta_sequence.size() > 0) {
              // Restart phase
              elbo_best = elbo;
              eta_best = eta;
            } else {
              // No more eta values to try, so use current eta if it
              // didn't diverge or fail if it did diverge
              if (elbo > elbo_init) {
                if (print_stream_)
                  *print_stream_ << "Success!"
                    << " Found best tuned hyperparameter"
                    << " [eta = " << eta_best
                    << "]" << std::endl << std::endl;
                eta_best = eta;
                do_more_tuning = false;
              } else {
                const char* name = "All proposed step-sizes";
                const char* msg1 = "failed. Your model may be either "
                  "severely ill-conditioned or misspecified.";
                stan::math::domain_error(function, name, "", msg1);
                return 0;
              }
            }
            // Reset
            params_prop = Q(model_.num_params_r());
          }
          variational = Q(cont_params_);
        }
        return eta_best;
      }

      /**
       * Runs stochastic gradient ascent with an adaptive stepsize.
       *
       * @param[in,out] variational This is the variational distribution we
       *     update with stochastic gradient ascent. The input is the initial
       *     variational distribution. The output is the variational
       *     distribution that maximizes the objective function (ELBO).
       * @param eta            eta parameter for stepsize scaling
       * @param tol_rel_obj    relative tolerance parameter for convergence
       * @param max_iterations max number of iterations to run algorithm
       */
      void stochastic_gradient_ascent(Q& variational,
                                      double eta,
                                      double tol_rel_obj,
                                      int max_iterations) const {
        static const char* function =
          "stan::variational::advi.stochastic_gradient_ascent";

        stan::math::check_nonnegative(function, "Eta stepsize", eta);
        stan::math::check_positive(function,
                                   "Relative objective function tolerance",
                                   tol_rel_obj);
        stan::math::check_positive(function,
                                   "Maximum iterations",
                                   max_iterations);

        // Gradient parameters
        Q elbo_grad = Q(model_.num_params_r());

        // Learning rate parameters
        Q params_prop = Q(model_.num_params_r());
        double tau = 1.0;
        double pre_factor  = 0.9;
        double post_factor = 0.1;
        double eta_scaled;

        // Initialize ELBO and convergence tracking variables
        double elbo(0.0);
        double elbo_best      = -std::numeric_limits<double>::max();
        double elbo_prev      = std::numeric_limits<double>::min();
        double delta_elbo     = std::numeric_limits<double>::max();
        double delta_elbo_ave = std::numeric_limits<double>::max();
        double delta_elbo_med = std::numeric_limits<double>::max();

        // Heuristic to estimate how far to look back in rolling window
        int cb_size = static_cast<int>(
                std::max(0.1*max_iterations/static_cast<double>(eval_elbo_),
                         2.0));
        boost::circular_buffer<double> elbo_diff(cb_size);

        // Timing variables
        clock_t start = clock();
        clock_t end;
        double delta_t;

        // Print main loop header
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

        // INTERMEDIATE SAMPLING OUTPUT
        int num_intermediate_samples = 25;
        std::fstream* intermediate_stream = 0;
        std::string intermediate_basename = "intermediate_variational_samples_";
        std::string intermediate_extension = ".csv";
        std::string intermediate_filename;
        std::vector<double> cont_vector(cont_params_.size());
        std::vector<int> disc_vector;

        // Main loop
        std::vector<double> print_vector;
        int iter_main = 1;
        bool do_more_iterations = true;
        while (do_more_iterations) {
          // Compute gradient of ELBO
          calc_ELBO_grad(variational, elbo_grad);

          // Update learning rate parameters
          if (iter_main == 1) {
            params_prop += elbo_grad.square();
          } else {
            params_prop = pre_factor * params_prop +
                          post_factor * elbo_grad.square();
          }
          eta_scaled = eta / sqrt(static_cast<double>(iter_main));

          // Stochastic gradient update
          variational += eta_scaled * elbo_grad / (tau + params_prop.sqrt());

          // Check for convergence every "eval_elbo_"th iteration
          if (iter_main % eval_elbo_ == 0) {
            elbo_prev = elbo;
            elbo = calc_ELBO(variational);
            if (elbo > elbo_best) {
              elbo_best = elbo;
            }
            delta_elbo = rel_difference_(elbo, elbo_prev);
            elbo_diff.push_back(delta_elbo);
            delta_elbo_ave = std::accumulate(
                               elbo_diff.begin(), elbo_diff.end(), 0.0)
                             / static_cast<double>(elbo_diff.size());
            delta_elbo_med = circ_buff_median_(elbo_diff);

            if (print_stream_) {
              *print_stream_
                        << "  "
                        << std::setw(4) << iter_main
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

              // INTERMEDIATE BUSINESS
              intermediate_filename = intermediate_basename
                + boost::lexical_cast<std::string>(iter_counter)
                + intermediate_extension;
              intermediate_stream = new std::fstream(intermediate_filename.c_str(),
                                                     std::fstream::out);

              for (int n = 0; n < num_intermediate_samples; ++n) {
                // cont_params_ = draw_posterior_sample(musigmatilde);
                variational.sample(rng_, cont_params_);
                for (int i = 0; i < cont_params_.size(); ++i)
                  cont_vector.at(i) = cont_params_(i);

                services::io::write_iteration(*intermediate_stream, model_, rng_,
                              0.0, cont_vector, disc_vector, print_stream_);
              }

              intermediate_stream->close();
              delete intermediate_stream;

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

            if (iter_main > 2*eval_elbo_) {
              if (delta_elbo_med > 0.5 || delta_elbo_ave > 0.5) {
                if (print_stream_)
                  *print_stream_ << "   MAY BE DIVERGING... INSPECT ELBO";
              }
            }

            if (print_stream_)
              *print_stream_ << std::endl;

            if (do_more_iterations == false &&
                rel_difference_(elbo, elbo_best) > 0.05) {
              if (print_stream_)
                *print_stream_
                  << "Informational Message: The ELBO at a previous "
                  << "iteration is larger than the ELBO upon "
                  << "convergence!"
                  << std::endl
                  << "This means that the variational approximation has "
                  << "not converged to the global optima."
                  << std::endl;
            }
          }

          if (iter_main == max_iterations) {
            if (print_stream_)
              *print_stream_
                << "Informational Message: The maximum number of "
                << "iterations is reached! The algorithm has not "
                << "converged."
                << std::endl
                << "Values from this variational approximation are not "
                << "guaranteed to be meaningful."
                << std::endl;
            do_more_iterations = false;
          }

          ++iter_main;
        }
      }

      /**
       * Runs the ADVI algorithm and writes to output.
       *
       * @param  eta            eta parameter for stepsize scaling
       * @param  tol_rel_obj    relative tolerance parameter for convergence
       * @param  max_iterations max number of iterations to run algorithm
       * @param  tuning_iter    number of iterations for hyperparameter tuning
       * @return stanserviceOK code if all goes well
       */
      int run(const std::string& eta, double tol_rel_obj,
              int max_iterations, int tuning_iter) const {
        if (diag_stream_) {
          *diag_stream_ << "iter,time_in_seconds,ELBO" << std::endl;
        }

        // initialize variational approximation
        Q variational = Q(cont_params_);

        // tune if eta is unspecified
        double eta_double(0);
        if (eta.compare("automatically tuned") == 0) {
          eta_double = tune(variational, tuning_iter);
          if (out_stream_) {
            *out_stream_ << "# Stepsize tuning complete." << std::endl
                         << "# eta = " << eta_double << std::endl;
          }
        } else {
          try {
            eta_double = boost::lexical_cast<double>(eta);
          } catch (std::exception& e) {
            std::stringstream eta_error_message;
            eta_error_message << "You specified the value eta = "
              << eta
              << " for the stepsize." << std::endl
              << "There is no valid conversion to a double." << std::endl
              << "Please check for typos. We recommend using "
              << "Stan's default automatic tuning procedure.";
            throw std::runtime_error(eta_error_message.str());
          }
          if (eta_double <= 0.0 || eta_double > 1.0){
            std::stringstream eta_error_message;
            eta_error_message << "You specified the value eta = "
              << eta
              << " for the stepsize." << std::endl
              << "This is outside of the allowable range 0 < eta <= 1."
              << std::endl
              << "Please check for typos. We recommend using "
              << "Stan's default automatic tuning procedure.";
            throw std::runtime_error(eta_error_message.str());
          }
        }

        // run inference algorithm
        stochastic_gradient_ascent(variational, eta_double, tol_rel_obj,
          max_iterations);

        // get mean of posterior approximation and write on first output line
        cont_params_ = variational.mean();
        // This is temporary as lp is not really helpful for variational
        // inference; furthermore it can be costly to compute.
        double lp = model_.template log_prob<false, true>(cont_params_,
          print_stream_);
        std::vector<double> cont_vector(cont_params_.size());
        for (int i = 0; i < cont_params_.size(); ++i)
          cont_vector.at(i) = cont_params_(i);
        std::vector<int> disc_vector;

        if (out_stream_) {
          services::io::write_iteration(*out_stream_, model_, rng_,
                                        lp, cont_vector, disc_vector,
                                        print_stream_);
        }

        // draw more samples from posterior and write on subsequent lines
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
            double lp = model_.template log_prob<false, true>(cont_params_,
              print_stream_);
            for (int i = 0; i < cont_params_.size(); ++i) {
              cont_vector.at(i) = cont_params_(i);
            }
            services::io::write_iteration(*out_stream_, model_, rng_,
                          lp, cont_vector, disc_vector, print_stream_);
          }

          if (print_stream_) {
            *print_stream_ << "DONE." << std::endl;
          }
        }

        return stan::services::error_codes::OK;
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

      double circ_buff_median_(const boost::circular_buffer<double>& cb) const {
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

      double rel_difference_(double prev, double curr) const {
        return std::abs(curr - prev) / std::abs(prev);
      }

      void write_error_msg_(std::ostream* error_msgs,
                            const std::exception& e) const {
        if (!error_msgs) {
          return;
        }

        *error_msgs
          << std::endl
          << "Informational Message: The current sample evaluation "
          << "of the ELBO is ignored because of the following issue:"
          << std::endl
          << e.what() << std::endl
          << "If this warning occurs often then your model may be "
          << "either severely ill-conditioned or misspecified."
          << std::endl;
      }
    };
  }  // variational
}  // stan

#endif
