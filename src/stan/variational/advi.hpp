#ifndef STAN_VARIATIONAL_ADVI_HPP
#define STAN_VARIATIONAL_ADVI_HPP

#include <stan/math/prim/scal/err/check_positive.hpp>

#include <stan/model/util.hpp>

#include <stan/services/io/write_iteration_csv.hpp>
#include <stan/services/io/write_iteration.hpp>
#include <stan/services/error_codes.hpp>

#include <stan/variational/advi_params_normal_fullrank.hpp>
#include <stan/variational/advi_params_normal_meanfield.hpp>

#include <stan/io/dump.hpp>

#include <boost/circular_buffer.hpp>
#include <ostream>
#include <limits>
#include <vector>
#include <numeric>
#include <algorithm>

namespace stan {

  namespace variational {

    /**
     * AUTOMATIC DIFFERENTIATION VARIATIONAL INFERENCE
     *
     * Calculates the "blackbox" Evidence Lower BOund (ELBO) by sampling
     * from the standard multivariate normal (for now), affine transform
     * the sample, and evaluating the log joint, adjusted by the entropy
     * term of the normal
     *
     * @tparam M                     class of model
     * @tparam BaseRNG               class of random number generator
     * @param  m                     stan model
     * @param  cont_params           initialization of continuous parameters
     * @param  n_monte_carlo_grad    number of samples for gradient computation
     * @param  n_monte_carlo_elbo    number of samples for ELBO computation
     * @param  eta_adagrad           eta parameter for adaGrad
     * @param  rng                   random number generator
     * @param  eval_elbo             evaluate ELBO at every "eval_elbo" iters
     * @param  n_posterior_samples   number of samples to draw from posterior
     * @param  print_stream          stream for convergence assessment output
     * @param  output_stream         stream for parameters output
     * @param  diagnostic_stream     stream for ELBO output
     */
    template <class M, class BaseRNG>
    class advi {
    public:
      advi(M& m,
           Eigen::VectorXd& cont_params,
           int n_monte_carlo_grad,
           int n_monte_carlo_elbo,
           double eta_adagrad,
           BaseRNG& rng,
           int eval_elbo,
           int n_posterior_samples,
           std::ostream* print_stream,
           std::ostream* output_stream,
           std::ostream* diagnostic_stream):
        model_(m),
        cont_params_(cont_params),
        rng_(rng),
        n_monte_carlo_grad_(n_monte_carlo_grad),
        n_monte_carlo_elbo_(n_monte_carlo_elbo),
        eta_adagrad_(eta_adagrad),
        eval_elbo_(eval_elbo),
        n_posterior_samples_(n_posterior_samples),
        print_stream_(print_stream),
        out_stream_(output_stream),
        diag_stream_(diagnostic_stream) {
        static const char* function =
          "stan::variational::advi";

        stan::math::check_positive(function,
                                 "Number of Monte Carlo samples for gradients",
                                 n_monte_carlo_grad_);

        stan::math::check_positive(function,
                                 "Number of Monte Carlo samples for ELBO",
                                 n_monte_carlo_elbo_);

        stan::math::check_positive(function,
                                 "Number of posterior samples for output",
                                 n_posterior_samples_);

        stan::math::check_positive(function, "Eta stepsize", eta_adagrad_);
        }

      /**
       * ELBO
       *
       * Calculates the "blackbox" Evidence Lower BOund (ELBO) by sampling from
       * the variational distribution and then evaluating the log joint,
       * adjusted by the entropy term of the variational distribution.
       *
       * @tparam  T                type of advi_params (meanfield or fullrank)
       * @param   advi_params      variational parameters class
       * @return                   evidence lower bound (elbo)
       */
      template <typename T>
      double calc_ELBO(const T& advi_params) {
        double elbo(0.0);
        int dim = advi_params.dimension();

        Eigen::VectorXd zeta = Eigen::VectorXd::Zero(dim);

        for (int i = 0; i < n_monte_carlo_elbo_; ++i) {
          // Draw from variational distribution
          zeta = advi_params.sample(rng_);

          // Accumulate log probability
          elbo += (model_.template log_prob<false, true>(zeta, print_stream_));
        }

        // Divide to get Monte Carlo integral estimate
        elbo /= n_monte_carlo_elbo_;

        // Add entropy term
        elbo += advi_params.entropy();

        return elbo;
      }

      /**
       * FULL-RANK ROBBINS-MONRO ADAGRAD
       *
       * Runs stochastic gradient ascent for some number of iterations
       *
       * @param muL            mean and cholesky factor of affine transform
       * @param tol_rel_obj    relative tolerance parameter for convergence
       * @param max_iterations max number of iterations to run algorithm
       */
      void robbins_monro_adagrad(advi_params_normal_fullrank& muL,
                                    double tol_rel_obj,
                                    int max_iterations) {
        static const char* function =
          "stan::variational::advi.robbins_monro_adagrad";

        stan::math::check_positive(function,
                                   "Relative objective function tolerance",
                                   tol_rel_obj);
        stan::math::check_positive(function,
                                   "Maximum iterations",
                                   max_iterations);

        // Gradients
        Eigen::VectorXd mu_grad = Eigen::VectorXd::Zero(model_.num_params_r());
        Eigen::MatrixXd L_grad  = Eigen::MatrixXd::Zero(model_.num_params_r(),
                                                        model_.num_params_r());

        // ADAgrad parameters
        double tau = 1.0;
        Eigen::VectorXd mu_s = Eigen::VectorXd::Zero(model_.num_params_r());
        Eigen::MatrixXd L_s  = Eigen::MatrixXd::Zero(model_.num_params_r(),
                                                     model_.num_params_r());

        // RMSprop window_size
        double window_size = 10.0;
        double post_factor = 1.0 / window_size;
        double pre_factor  = 1.0 - post_factor;

        // Initialize ELBO and convergence tracking variables
        double elbo(0.0);
        double elbo_prev = std::numeric_limits<double>::min();
        double delta_elbo = std::numeric_limits<double>::max();
        double delta_elbo_ave = std::numeric_limits<double>::max();
        double delta_elbo_med = std::numeric_limits<double>::max();

        // Heuristic to estimate how far to look back in rolling window
        int cb_size = static_cast<int>(
                std::max(0.1*max_iterations/static_cast<double>(eval_elbo_),
                         1.0));
        boost::circular_buffer<double> cb(cb_size);

        // Print stuff
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
        int iter_counter = 0;
        while (do_more_iterations) {
          // Compute gradient using Monte Carlo integration
          muL.calc_grad(mu_grad, L_grad,
                        model_, cont_params_, n_monte_carlo_grad_, rng_,
                        print_stream_);

          if (iter_counter == 0) {
            mu_s.array() += mu_grad.array().square();
            L_s.array() += L_grad.array().square();
          } else {
            // RMSprop moving average weighting
            mu_s.array() = pre_factor * mu_s.array()
                           + post_factor *
                           mu_grad.array().square();
            L_s.array()  = pre_factor * L_s.array()
                           + post_factor *
                           L_grad.array().square();
          }

          // Take ADAgrad or rmsprop step
          muL.set_mu(muL.mu().array() + eta_adagrad_ * mu_grad.array()
             / (tau + mu_s.array().sqrt()));
          muL.set_L_chol(muL.L_chol().array() + eta_adagrad_ * L_grad.array()
             / (tau + L_s.array().sqrt()));

          // Check for convergence every "eval_elbo_"th iteration
          if (iter_counter % eval_elbo_ == 0) {
            elbo_prev = elbo;
            elbo = calc_ELBO(muL);
            delta_elbo = rel_decrease(elbo, elbo_prev);
            cb.push_back(delta_elbo);
            delta_elbo_ave = std::accumulate(cb.begin(), cb.end(), 0.0)
                             / static_cast<double>(cb.size());
            delta_elbo_med = circ_buff_median(cb);
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

            if (iter_counter > 100) {
              if (delta_elbo_med > 0.5 || delta_elbo_ave > 0.5) {
                if (print_stream_)
                  *print_stream_ << "   MAY BE DIVERGING... INSPECT ELBO";
              }
            }

            if (print_stream_)
              *print_stream_ << std::endl;
          }

          // Check for max iterations
          if (iter_counter == max_iterations) {
            if (print_stream_)
              *print_stream_ << "MAX ITERATIONS" << std::endl;
            do_more_iterations = false;
          }

          ++iter_counter;
        }
      }

      /**
       * MEAN-FIELD ROBBINS-MONRO ADAGRAD
       *
       * Runs stochastic gradient ascent for some number of iterations
       *
       * @param muomega         mean and log-std vector of affine transform
       * @param tol_rel_obj     relative tolerance parameter for convergence
       * @param max_iterations  max number of iterations to run algorithm
       */
      void robbins_monro_adagrad(advi_params_normal_meanfield& muomega,
                                    double tol_rel_obj,
                                    int max_iterations) {
        static const char* function =
          "stan::variational::advi.robbins_monro_adagrad";

        stan::math::check_positive(function,
                                   "Relative objective function tolerance",
                                   tol_rel_obj);
        stan::math::check_positive(function,
                                   "Maximum iterations",
                                   max_iterations);

        // Gradients
        Eigen::VectorXd mu_grad =
          Eigen::VectorXd::Zero(model_.num_params_r());
        Eigen::VectorXd omega_grad =
          Eigen::VectorXd::Zero(model_.num_params_r());

        // ADAgrad parameters
        double tau = 1.0;
        Eigen::VectorXd mu_s =
          Eigen::VectorXd::Zero(model_.num_params_r());
        Eigen::VectorXd omega_s =
          Eigen::VectorXd::Zero(model_.num_params_r());

        // RMSprop window_size
        double window_size = 10.0;
        double post_factor = 1.0 / window_size;
        double pre_factor  = 1.0 - post_factor;

        // Initialize ELBO and convergence tracking variables
        double elbo(0.0);
        double elbo_prev      = std::numeric_limits<double>::min();
        double delta_elbo     = std::numeric_limits<double>::max();
        double delta_elbo_ave = std::numeric_limits<double>::max();
        double delta_elbo_med = std::numeric_limits<double>::max();

        // Heuristic to estimate how far to look back in rolling window
        int cb_size = static_cast<int>(
                std::max(0.1*max_iterations/static_cast<double>(eval_elbo_),
                         1.0));
        boost::circular_buffer<double> cb(cb_size);

        // Print stuff
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
        int iter_counter = 0;
        while (do_more_iterations) {
          // Compute gradient using Monte Carlo integration
          muomega.calc_grad(mu_grad, omega_grad,
                            model_, cont_params_, n_monte_carlo_grad_, rng_,
                            print_stream_);

          if (iter_counter == 0) {
            mu_s.array() += mu_grad.array().square();
            omega_s.array() += omega_grad.array().square();
          } else {
            // RMSprop moving average weighting
            mu_s.array() = pre_factor * mu_s.array()
                         + post_factor * mu_grad.array().square();
            omega_s.array() = pre_factor * omega_s.array()
                            + post_factor * omega_grad.array().square();
          }

          // Take ADAgrad or rmsprop step
          muomega.set_mu(muomega.mu().array()
            + eta_adagrad_ * mu_grad.array()
            / (tau + mu_s.array().sqrt()) );
          muomega.set_omega(muomega.omega().array()
            + eta_adagrad_ * omega_grad.array()
            / (tau + omega_s.array().sqrt()) );

          // Check for convergence every "eval_elbo_"th iteration
          if (iter_counter % eval_elbo_ == 0) {
            elbo_prev  = elbo;
            elbo       = calc_ELBO(muomega);
            delta_elbo = rel_decrease(elbo, elbo_prev);
            cb.push_back(delta_elbo);
            delta_elbo_ave = std::accumulate(cb.begin(), cb.end(), 0.0)
                             / static_cast<double>(cb.size());
            delta_elbo_med = circ_buff_median(cb);
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

            if (iter_counter > 100) {
              if (delta_elbo_med > 0.5 || delta_elbo_ave > 0.5) {
                if (print_stream_)
                  *print_stream_ << "   MAY BE DIVERGING... INSPECT ELBO";
              }
            }

            if (print_stream_)
              *print_stream_ << std::endl;
          }

          // Check for max iterations
          if (iter_counter == max_iterations) {
            if (print_stream_)
              *print_stream_ << "MAX ITERATIONS REACHED" << std::endl;
            do_more_iterations = false;
          }

          ++iter_counter;
        }
      }

      int run_fullrank(double tol_rel_obj, int max_iterations) {
        if (print_stream_) {
          *print_stream_
            << "This is Automatic Differentiation Variational Inference"
            << " (full-rank)." << std::endl << std::endl;
        }

        if (diag_stream_) {
          *diag_stream_ << "iter,time_in_seconds,ELBO" << std::endl;
        }

        // initialize variational approximation
        Eigen::VectorXd mu = cont_params_;
        Eigen::MatrixXd L  = Eigen::MatrixXd::Identity(model_.num_params_r(),
                                                       model_.num_params_r());
        advi_params_normal_fullrank muL = advi_params_normal_fullrank(mu, L);

        // run inference algorithm
        robbins_monro_adagrad(muL, tol_rel_obj, max_iterations);

        // get mean of posterior approximation and write on first output line
        cont_params_ = muL.mu();
        std::vector<double> cont_vector(cont_params_.size());
        for (int i = 0; i < cont_params_.size(); ++i)
          cont_vector.at(i) = cont_params_(i);
        std::vector<int> disc_vector;

        if (out_stream_) {
          services::io::write_iteration(*out_stream_, model_, rng_,
                          0.0, cont_vector, disc_vector);
        }

        // draw more samples from posterior and write on subsequent lines
        if (out_stream_) {
          for (int n = 0; n < n_posterior_samples_; ++n) {
            cont_params_ = muL.sample(rng_);
            for (int i = 0; i < cont_params_.size(); ++i)
              cont_vector.at(i) = cont_params_(i);
            services::io::write_iteration(*out_stream_, model_, rng_,
                          0.0, cont_vector, disc_vector);
          }
        }

        return stan::services::error_codes::OK;
      }

      int run_meanfield(double tol_rel_obj, int max_iterations) {
        if (print_stream_) {
          *print_stream_
            << "This is Automatic Differentiation Variational Inference"
            << " (mean-field)." << std::endl << std::endl;
        }

        if (diag_stream_) {
          *diag_stream_ << "iter,time_in_seconds,ELBO" << std::endl;
        }

        // initialize variational approximation
        Eigen::VectorXd mu     = cont_params_;
        Eigen::MatrixXd omega  = Eigen::VectorXd::Constant(
                                                model_.num_params_r(), 0.0);
                                                // initializing omega = 0
                                                // means sigma = 1
        advi_params_normal_meanfield muomega =
          advi_params_normal_meanfield(mu, omega);

        // run inference algorithm
        robbins_monro_adagrad(muomega, tol_rel_obj, max_iterations);

        // get mean of posterior approximation and write on first output line
        cont_params_ = muomega.mu();
        std::vector<double> cont_vector(cont_params_.size());
        for (int i = 0; i < cont_params_.size(); ++i)
          cont_vector.at(i) = cont_params_(i);
        std::vector<int> disc_vector;

        if (out_stream_) {
          services::io::write_iteration(*out_stream_, model_, rng_,
                          0.0, cont_vector, disc_vector);
        }

        // draw more samples from posterior and write on subsequent lines
        if (out_stream_) {
          for (int n = 0; n < n_posterior_samples_; ++n) {
            cont_params_ = muomega.sample(rng_);
            for (int i = 0; i < cont_params_.size(); ++i)
              cont_vector.at(i) = cont_params_(i);
            services::io::write_iteration(*out_stream_, model_, rng_,
                          0.0, cont_vector, disc_vector);
          }
        }

        return stan::services::error_codes::OK;
      }

      // Helper function: compute the median of a circular buffer
      double circ_buff_median(const boost::circular_buffer<double>& cb) {
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
      M& model_;
      Eigen::VectorXd& cont_params_;
      BaseRNG& rng_;
      int n_monte_carlo_grad_;
      int n_monte_carlo_elbo_;
      double eta_adagrad_;
      int eval_elbo_;
      int n_posterior_samples_;
      std::ostream* print_stream_;
      std::ostream* out_stream_;
      std::ostream* diag_stream_;
    };
  }  // variational
}  // stan

#endif

