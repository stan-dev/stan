#ifndef STAN_VARIATIONAL_ADVI_HPP
#define STAN_VARIATIONAL_ADVI_HPP

#include <stan/math/prim.hpp>
#include <stan/math/prim/scal/err/check_positive.hpp>

#include <stan/model/util.hpp>

#include <stan/services/io/write_iteration_csv.hpp>
#include <stan/services/io/write_iteration.hpp>

#include <stan/variational/advi_params_fullrank.hpp>
#include <stan/variational/advi_params_meanfield.hpp>

#include <stan/io/dump.hpp>

#include <boost/circular_buffer.hpp>
#include <ostream>
#include <limits>
#include <vector>
#include <numeric>
#include <algorithm>

namespace stan {

  namespace variational {

    template <class M, class BaseRNG>
    class advi {
    public:
      advi(M& m,
           Eigen::VectorXd& cont_params,
           int n_monte_carlo_grad,
           int n_monte_carlo_elbo,
           double eta_stepsize,
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
        eta_stepsize_(eta_stepsize),
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

        stan::math::check_positive(function, "Eta stepsize", eta_stepsize_);
        }

      /**
       * ELBO
       *
       * Calculates the "blackbox" Evidence Lower BOund (ELBO) by sampling
       * from the standard multivariate normal (for now), affine transform
       * the sample, and evaluating the log joint, adjusted by the entropy
       * term of the normal
       *
       * @param   advi_params      variational parameters class
       * @return                   evidence lower bound (elbo)
       */
      template <typename T>
      double calc_ELBO(const T& advi_params) {
        double elbo(0.0);
        int dim = advi_params.dimension();

        Eigen::VectorXd z_check   = Eigen::VectorXd::Zero(dim);
        Eigen::VectorXd z_tilde   = Eigen::VectorXd::Zero(dim);

        for (int i = 0; i < n_monte_carlo_elbo_; ++i) {
          // Draw from standard normal and transform to unconstrained space
          for (int d = 0; d < dim; ++d) {
            z_check(d) = stan::prob::normal_rng(0, 1, rng_);
          }
          z_tilde = advi_params.loc_scale_transform(z_check);

          // Accumulate log probability
          elbo += (model_.template
                   log_prob<false, true>(z_tilde, print_stream_));
        }
        // Divide to get Monte Carlo integral estimate
        elbo /= static_cast<double>(n_monte_carlo_elbo_);

        // Add entropy term
        elbo += advi_params.entropy();

        return elbo;
      }

      /**
       * Draws samples from the (converged) posterior, which is a multivariate
       * Gaussian distribution. The posterior can be either a fullrank Gaussian
       * or a mean-field (diagonal) Gaussian.
       *
       * @param   advi_params      variational parameters class
       * @return                   a sample from the posterior distribution
       */
      template <typename T>
      Eigen::VectorXd draw_posterior_sample(const T& advi_params) {
        int dim = advi_params.dimension();
        Eigen::VectorXd z_check = Eigen::VectorXd::Zero(dim);

        // Draw from standard normal and transform to unconstrained space
        for (int d = 0; d < dim; ++d) {
          z_check(d) = stan::prob::normal_rng(0, 1, rng_);
        }

        return advi_params.loc_scale_transform(z_check);
      }

      /**
       * FULL-RANK GRADIENTS
       *
       * Calculates the "blackbox" gradient with respect to BOTH the location
       * vector (mu) and the cholesky factor of the scale matrix (L) in
       * parallel. It uses the same gradient computed from a set of Monte Carlo
       * samples
       *
       * @param muL     mean and cholesky factor of affine transform
       * @param mu_grad gradient of location vector parameter
       * @param L_grad  gradient of scale matrix parameter
       */
      void calc_combined_grad(const advi_params_fullrank& muL,
                              Eigen::VectorXd& mu_grad,
                              Eigen::MatrixXd& L_grad) {
        static const char* function =
          "stan::variational::advi.calc_combined_grad";

        int dim       = muL.dimension();
        double tmp_lp = 0.0;

        stan::math::check_size_match(function,
                        "Dimension of muL", muL.dimension(),
                        "Dimension of variables in model", cont_params_.size());
        stan::math::check_size_match(function,
                        "Dimension of mu grad vector", mu_grad.size(),
                        "Dimension of mean vector in variational q", dim);
        stan::math::check_square(function, "Scale matrix", L_grad);
        stan::math::check_size_match(function,
                        "Dimension of scale matrix", L_grad.rows(),
                        "Dimension of mean vector in variational q", dim);

        // Initialize everything to zero
        mu_grad = Eigen::VectorXd::Zero(dim);
        L_grad  = Eigen::MatrixXd::Zero(dim, dim);
        Eigen::VectorXd tmp_mu_grad = Eigen::VectorXd::Zero(dim);
        Eigen::VectorXd z_check = Eigen::VectorXd::Zero(dim);
        Eigen::VectorXd z_tilde = Eigen::VectorXd::Zero(dim);

        // Naive Monte Carlo integration
        for (int i = 0; i < n_monte_carlo_grad_; ++i) {
          // Draw from standard normal and transform to unconstrained space
          for (int d = 0; d < dim; ++d) {
            z_check(d) = stan::prob::normal_rng(0, 1, rng_);
          }
          z_tilde = muL.loc_scale_transform(z_check);

          // Compute gradient step in unconstrained space
          stan::model::gradient(model_, z_tilde, tmp_lp, tmp_mu_grad,
                                print_stream_);

          // Update mu
          mu_grad += tmp_mu_grad;

          // Update L (lower triangular)
          for (int ii = 0; ii < dim; ++ii) {
            for (int jj = 0; jj <= ii; ++jj) {
              L_grad(ii, jj) += tmp_mu_grad(ii) * z_check(jj);
            }
          }
        }
        mu_grad /= static_cast<double>(n_monte_carlo_grad_);
        L_grad  /= static_cast<double>(n_monte_carlo_grad_);

        // Add gradient of entropy term
        L_grad.diagonal().array() += muL.L_chol().diagonal().array().inverse();
      }


      /**
       * MEAN-FIELD GRADIENTS
       *
       * Calculates the "blackbox" gradient with respect to BOTH the location
       * vector (mu) and the variance vector (sigma^2) in parallel.
       * It uses the same gradient computed from a set of Monte Carlo
       * samples
       *
       * @param musigmatilde      mean and log-std vector of affine transform
       * @param mu_grad           gradient of mean vector parameter
       * @param sigma_tilde_grad  gradient of log-std vector parameter
       */
      void calc_combined_grad(const advi_params_meanfield& musigmatilde,
                              Eigen::VectorXd& mu_grad,
                              Eigen::VectorXd& sigma_tilde_grad) {
        static const char* function =
          "stan::variational::advi.calc_combined_grad";

        int dim       = musigmatilde.dimension();
        double tmp_lp = 0.0;

        stan::math::check_size_match(function,
                              "Dimension of mu grad vector", mu_grad.size(),
                              "Dimension of mean vector in variational q", dim);
        stan::math::check_size_match(function,
                "Dimension of sigma_tilde grad vector", sigma_tilde_grad.size(),
                "Dimension of mean vector in variational q", dim);
        stan::math::check_size_match(function,
                        "Dimension of musigmatilde", dim,
                        "Dimension of variables in model", cont_params_.size());


        // Initialize everything to zero
        mu_grad          = Eigen::VectorXd::Zero(dim);
        sigma_tilde_grad = Eigen::VectorXd::Zero(dim);

        Eigen::VectorXd tmp_mu_grad = Eigen::VectorXd::Zero(dim);

        Eigen::VectorXd z_check = Eigen::VectorXd::Zero(dim);
        Eigen::VectorXd z_tilde = Eigen::VectorXd::Zero(dim);

        // Naive Monte Carlo integration
        for (int i = 0; i < n_monte_carlo_grad_; ++i) {
          // Draw from standard normal and transform to unconstrained space
          for (int d = 0; d < dim; ++d) {
            z_check(d) = stan::prob::normal_rng(0, 1, rng_);
          }
          z_tilde = musigmatilde.loc_scale_transform(z_check);

          stan::math::check_not_nan(function, "z_tilde", z_tilde);

          // Compute gradient step in unconstrained space
          stan::model::gradient(model_, z_tilde, tmp_lp, tmp_mu_grad,
                                print_stream_);

          // Update mu
          mu_grad.array() = mu_grad.array() + tmp_mu_grad.array();

          // Update sigma_tilde
          sigma_tilde_grad.array() = sigma_tilde_grad.array()
            + tmp_mu_grad.array().cwiseProduct(z_check.array());
        }
        mu_grad           /= static_cast<double>(n_monte_carlo_grad_);
        sigma_tilde_grad  /= static_cast<double>(n_monte_carlo_grad_);

        // multiply by exp(sigma_tilde)
        sigma_tilde_grad.array() =
          sigma_tilde_grad.array().cwiseProduct(
                                      musigmatilde.sigma_tilde().array().exp());

        // Add gradient of entropy term (just equal to element-wise 1 here)
        sigma_tilde_grad.array() += 1.0;
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
      void robbins_monro_adagrad(advi_params_fullrank& muL,
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
        double window_size = 100.0;
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
        *print_stream_ << "  iter"
                       << "       ELBO"
                       << "   delta_ELBO_mean"
                       << "   delta_ELBO_med"
                       << "   notes "
                       << std::endl;

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
          calc_combined_grad(muL, mu_grad, L_grad);

          // Accumulate S vector for ADAgrad
          mu_s.array() += mu_grad.array().square();
          L_s.array()  += L_grad.array().square();

          // RMSprop moving average weighting
          mu_s.array() = pre_factor * mu_s.array()
                       + post_factor *
                       mu_grad.array().square();
          L_s.array()  = pre_factor * L_s.array()
                       + post_factor *
                       L_grad.array().square();

          // Take ADAgrad or rmsprop step
          muL.set_mu(muL.mu().array()
             + eta_stepsize_ * mu_grad.array()
             / (tau + mu_s.array().sqrt()));
          muL.set_L_chol(muL.L_chol().array()
             + eta_stepsize_ * L_grad.array()
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
              *print_stream_ << "   MEAN ELBO CONVERGED";
              do_more_iterations = false;
            }

            if (delta_elbo_med < tol_rel_obj) {
              *print_stream_ << "   MEDIAN ELBO CONVERGED";
              do_more_iterations = false;
            }

            if (iter_counter > 100) {
              if (delta_elbo_med > 0.5 || delta_elbo_ave > 0.5) {
                *print_stream_ << "   MAY BE DIVERGING... INSPECT ELBO";
              }
            }

            *print_stream_ << std::endl;
          }

          // Check for max iterations
          if (iter_counter == max_iterations) {
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
       * @param musigmatilde    mean and log-std vector of affine transform
       * @param tol_rel_obj   relative tolerance parameter for convergence
       * @param max_iterations  max number of iterations to run algorithm
       */
      void robbins_monro_adagrad(advi_params_meanfield& musigmatilde,
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
        Eigen::VectorXd sigma_tilde_grad =
          Eigen::VectorXd::Zero(model_.num_params_r());

        // ADAgrad parameters
        double tau = 1.0;
        Eigen::VectorXd mu_s =
          Eigen::VectorXd::Zero(model_.num_params_r());
        Eigen::VectorXd sigma_tilde_s =
          Eigen::VectorXd::Zero(model_.num_params_r());

        // RMSprop window_size
        double window_size = 100.0;
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
        *print_stream_ << "  iter"
                       << "       ELBO"
                       << "   delta_ELBO_mean"
                       << "   delta_ELBO_med"
                       << "   notes "
                       << std::endl;

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
          calc_combined_grad(musigmatilde, mu_grad, sigma_tilde_grad);

          // Accumulate S vector for ADAgrad
          mu_s.array()           += mu_grad.array().square();
          sigma_tilde_s.array()  += sigma_tilde_grad.array().square();

          // RMSprop moving average weighting
          mu_s.array() = pre_factor * mu_s.array()
                       + post_factor * mu_grad.array().square();
          sigma_tilde_s.array() = pre_factor * sigma_tilde_s.array()
                                + post_factor
                                  * sigma_tilde_grad.array().square();

          // Take ADAgrad or rmsprop step
          musigmatilde.set_mu(
            musigmatilde.mu().array()
            + eta_stepsize_ * mu_grad.array()
            / (tau + mu_s.array().sqrt()));
          musigmatilde.set_sigma_tilde(
            musigmatilde.sigma_tilde().array()
            + eta_stepsize_ * sigma_tilde_grad.array()
            / (tau + sigma_tilde_s.array().sqrt()));

          // Check for convergence every "eval_elbo_"th iteration
          if (iter_counter % eval_elbo_ == 0) {
            elbo_prev = elbo;
            elbo = calc_ELBO(musigmatilde);
            delta_elbo = rel_decrease(elbo, elbo_prev);
            cb.push_back(delta_elbo);
            delta_elbo_ave = std::accumulate(cb.begin(), cb.end(), 0.0)
                             / static_cast<double>(cb.size());
            delta_elbo_med = circ_buff_median(cb);
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
              *print_stream_ << "   MEAN ELBO CONVERGED";
              do_more_iterations = false;
            }

            if (delta_elbo_med < tol_rel_obj) {
              *print_stream_ << "   MEDIAN ELBO CONVERGED";
              do_more_iterations = false;
            }

            if (iter_counter > 100) {
              if (delta_elbo_med > 0.5 || delta_elbo_ave > 0.5) {
                *print_stream_ << "   MAY BE DIVERGING... INSPECT ELBO";
              }
            }

            *print_stream_ << std::endl;
          }

          // Check for max iterations
          if (iter_counter == max_iterations) {
            *print_stream_ << "MAX ITERATIONS REACHED" << std::endl;
            do_more_iterations = false;
          }

          ++iter_counter;
        }
      }

      void run_fullrank(double tol_rel_obj, int max_iterations) {
        *print_stream_
        << "This is stan::variational::advi::run_fullrank()" << std::endl;

        // initialize variational approximation
        Eigen::VectorXd mu = cont_params_;
        Eigen::MatrixXd L  = Eigen::MatrixXd::Identity(model_.num_params_r(),
                                                       model_.num_params_r());
        advi_params_fullrank muL = advi_params_fullrank(mu, L);

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
            cont_params_ = draw_posterior_sample(muL);
            for (int i = 0; i < cont_params_.size(); ++i)
              cont_vector.at(i) = cont_params_(i);

            services::io::write_iteration(*out_stream_, model_, rng_,
                          0.0, cont_vector, disc_vector);
          }
        }

        return;
      }

      void run_meanfield(double tol_rel_obj, int max_iterations) {
        *print_stream_
        << "This is stan::variational::advi::run_meanfield()" << std::endl;

        // initialize variational approximation
        Eigen::VectorXd mu           = cont_params_;
        Eigen::MatrixXd sigma_tilde  = Eigen::VectorXd::Constant(
                                                model_.num_params_r(),
                                                0.0);
                                                // initializing sigma_tilde = 0
                                                // means sigma = 1
        advi_params_meanfield musigmatilde =
          advi_params_meanfield(mu, sigma_tilde);

        // run inference algorithm
        robbins_monro_adagrad(musigmatilde, tol_rel_obj, max_iterations);

        // get mean of posterior approximation and write on first output line
        cont_params_ = musigmatilde.mu();
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
            cont_params_ = draw_posterior_sample(musigmatilde);
            for (int i = 0; i < cont_params_.size(); ++i)
              cont_vector.at(i) = cont_params_(i);

            services::io::write_iteration(*out_stream_, model_, rng_,
                          0.0, cont_vector, disc_vector);
          }
        }

        return;
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
      double eta_stepsize_;
      int eval_elbo_;
      int n_posterior_samples_;
      std::ostream* print_stream_;
      std::ostream* out_stream_;
      std::ostream* diag_stream_;
    };
  }  // variational
}  // stan

#endif

