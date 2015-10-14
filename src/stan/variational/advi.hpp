#ifndef STAN_VARIATIONAL_ADVI_HPP
#define STAN_VARIATIONAL_ADVI_HPP

#include <stan/math.hpp>
#include <stan/io/dump.hpp>
#include <stan/model/util.hpp>
#include <stan/services/io/write_iteration_csv.hpp>
#include <stan/services/io/write_iteration.hpp>
#include <stan/services/error_codes.hpp>
#include <stan/variational/families/normal_fullrank.hpp>
#include <stan/variational/families/normal_meanfield.hpp>
#include <boost/circular_buffer.hpp>
#include <algorithm>
#include <limits>
#include <numeric>
#include <ostream>
#include <vector>

namespace stan {

  namespace variational {

    template <class M, class Q, class BaseRNG>
    class advi {
    public:
      advi(M& m,
           Eigen::VectorXd& cont_params,
           BaseRNG& rng,
           int n_monte_carlo_grad,
           int n_monte_carlo_elbo,
           double eta,
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
        eta_(eta),
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
        stan::math::check_positive(function, "Eta stepsize", eta_);
        }

      /**
       * Calculates the Evidence Lower BOund (ELBO) by sampling from
       * the variational distribution and then evaluating the log joint,
       * adjusted by the entropy term of the variational distribution.
       *
       * @param variational variational distribution
       * @return evidence lower bound (elbo)
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
       * @param elbo_grad gradient of ELBO with respect to variational
       *                  parameters
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

        variational.calc_grad(elbo_grad, model_, cont_params_,
          n_monte_carlo_grad_, rng_, print_stream_);
      }

      /**
       * Runs stochastic gradient descent.
       *
       * @param[in,out] variational variational distribution
       * @param tol_rel_obj relative tolerance for convergence
       * @param max_iterations max number of iterations to run algorithm
       * @return stan::services::error_codes::OK
       */
      int stochastic_gradient_descent(Q& variational,
                                       double tol_rel_obj,
                                       int max_iterations) const {
        static const char* function =
          "stan::variational::advi::stochastic_gradient_descent";

        stan::math::check_positive(function,
                                   "Relative objective function tolerance",
                                   tol_rel_obj);
        stan::math::check_positive(function,
                                   "Maximum iterations",
                                   max_iterations);

        // Gradient parameters
        Q elbo_grad = Q(model_.num_params_r());

        // Learning rate parameters
        double tau = 1.0;
        Q params_prop = Q(model_.num_params_r());
        double pre_factor  = 0.9;
        double post_factor = 0.1;

        // Initialize ELBO and convergence tracking variables
        double elbo(0.0);
        double elbo_prev      = std::numeric_limits<double>::min();
        double delta_elbo     = std::numeric_limits<double>::max();
        double delta_elbo_ave = std::numeric_limits<double>::max();
        double delta_elbo_med = std::numeric_limits<double>::max();
        int cb_size = static_cast<int>(
                std::max(0.1*max_iterations/static_cast<double>(eval_elbo_),
                         1.0));
        boost::circular_buffer<double> elbo_diff(cb_size);

        // Print main loop header
        if (print_stream_) {
          *print_stream_ << "  iter"
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
        for (int iter_sgd = 1; do_more_iterations; ++iter_sgd) {
          // Compute gradient of ELBO
          calc_ELBO_grad(variational, elbo_grad);

          // Update learning rate parameters
          if (iter_sgd == 1) {
            params_prop += elbo_grad.square();
          } else {
            params_prop = pre_factor * params_prop +
                          post_factor * elbo_grad.square();
          }

          // Stochastic gradient update
          variational += eta_ * elbo_grad / (tau + params_prop.sqrt());

          // Check for convergence every "eval_elbo_"th iteration
          if (iter_sgd % eval_elbo_ == 0) {
            elbo_prev = elbo;
            elbo = calc_ELBO(variational);
            delta_elbo = rel_decrease(elbo, elbo_prev);
            elbo_diff.push_back(delta_elbo);
            delta_elbo_ave = std::accumulate(elbo_diff.begin(),
                                             elbo_diff.end(),
                                             0.0) /
                             static_cast<double>(elbo_diff.size());
            delta_elbo_med = circ_buff_median(elbo_diff);
            if (print_stream_) {
              *print_stream_
                        << "  "
                        << std::setw(4) << iter_sgd
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
                                                iter_sgd,
                                                print_vector);
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

            if (iter_sgd > 100) {
              if (delta_elbo_med > 0.5 || delta_elbo_ave > 0.5) {
                if (print_stream_)
                  *print_stream_ << "   MAY BE DIVERGING... INSPECT ELBO";
              }
            }

            if (print_stream_)
              *print_stream_ << std::endl;
          }

          // Check for max iterations
          if (iter_sgd == max_iterations) {
            if (print_stream_)
              *print_stream_ << "MAX ITERATIONS REACHED" << std::endl;
            do_more_iterations = false;
          }
        }
        return stan::services::error_codes::OK;
      }

      /**
       * Pre-processing steps.
       */
      void pre_process() const {
        if (diag_stream_) {
          *diag_stream_ << "iter,time_in_seconds,ELBO" << std::endl;
        }
      }

      /**
       * Post-processing steps, writing to output.
       *
       * @param variational variational distribution
       */
      void post_process(const Q& variational) const {
        // Write mean of posterior approximation to first line
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

        // Draw approximate posterior samples and write to subsequent lines
        if (out_stream_) {
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
        }
      }

      /**
       * Runs ADVI.
       *
       * @param tol_rel_obj    relative tolerance parameter for convergence
       * @param max_iterations max number of iterations to run algorithm
       * @return stan::services::error_codes::OK
       */
      int run(double tol_rel_obj, int max_iterations) const {
        pre_process();
        Q variational = Q(cont_params_);
        stochastic_gradient_descent(variational, tol_rel_obj, max_iterations);
        post_process(variational);
        return stan::services::error_codes::OK;
      }

      // Helper function: compute the median of a circular buffer
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

      // Helper function: compute relative decrease between two doubles
      double rel_decrease(double prev, double curr) const {
        return std::abs(curr - prev) / std::abs(prev);
      }

    protected:
      M& model_;                     // model
      Eigen::VectorXd& cont_params_; // parameters
      BaseRNG& rng_;                 // random number generator
      int n_monte_carlo_grad_;       // # of samples for gradient computation
      int n_monte_carlo_elbo_;       // # of samples for ELBO computation
      double eta_;                   // stepsize scaling in learning rate
      int eval_elbo_;                // evaluate ELBO every "eval_elbo" iters
      int n_posterior_samples_;      // # of samples to draw after convergence
      std::ostream* print_stream_;   // progress output
      std::ostream* out_stream_;     // final output
      std::ostream* diag_stream_;    // diagnostic output

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
