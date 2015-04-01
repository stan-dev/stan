#ifndef STAN__VARIATIONAL__ADVI__HPP
#define STAN__VARIATIONAL__ADVI__HPP

#include <boost/circular_buffer.hpp>
#include <ostream>
#include <numeric>
#include <algorithm>

#include <stan/math/prim.hpp>
#include <stan/model/util.hpp>

#include <stan/services/io/write_iteration_csv.hpp>

#include <stan/variational/advi_params_fullrank.hpp>
#include <stan/variational/advi_params_meanfield.hpp>

#include <stan/io/dump.hpp>

namespace stan {

  namespace variational {

    template <class M, class BaseRNG>
    class advi {

    public:

      advi(M& m,
           Eigen::VectorXd& cont_params,
           double& elbo,
           int& n_monte_carlo_grad,
           int& n_monte_carlo_elbo,
           double& eta_stepsize,
           BaseRNG& rng,
           std::ostream* output_stream,
           int& refresh,
           std::ostream* diagnostic_stream):
        model_(m),
        cont_params_(cont_params),
        elbo_(elbo),
        rng_(rng),
        n_monte_carlo_grad_(n_monte_carlo_grad),
        n_monte_carlo_elbo_(n_monte_carlo_elbo),
        eta_stepsize_(eta_stepsize),
        refresh_(refresh),
        out_stream_(output_stream),
        diag_stream_(diagnostic_stream) {};

      virtual ~advi() {};


      /**
       * ELBO
       *
       * Calculates the "blackbox" Evidence Lower BOund (ELBO) by sampling
       * from the standard multivariate normal (for now), affine transform
       * the sample, and evaluating the log joint, adjusted by the entropy
       * term of the normal
       *
       * @param   advi_params        variational parameters class
       * @param   constant_factor  difference between log_prob agrad and double
       * @return                   evidence lower bound (elbo)
       */
      template <typename T>
      double calc_ELBO(T const& advi_params,
                       double constant_factor) {
        double elbo(0.0);
        int dim = advi_params.dimension();

        Eigen::VectorXd z_check   = Eigen::VectorXd::Zero(dim);
        Eigen::VectorXd z_tilde   = Eigen::VectorXd::Zero(dim);

        for (int i = 0; i < n_monte_carlo_elbo_; ++i) {
          // Draw from standard normal and transform to unconstrained space
          for (int d = 0; d < dim; ++d) {
            z_check(d) = stan::prob::normal_rng(0,1,rng_);
          }
          z_tilde = advi_params.to_unconstrained(z_check);

          // Accumulate log probability
          elbo += (model_.template log_prob<false,true>(z_tilde, &std::cout))
                   + constant_factor;
        }
        // Divide to get Monte Carlo integral estimate
        elbo /= static_cast<double>(n_monte_carlo_elbo_);

        // Add entropy term
        elbo += advi_params.entropy();

        return elbo;
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
      void calc_combined_grad(
        advi_params_fullrank const& muL,
        Eigen::VectorXd& mu_grad,
        Eigen::MatrixXd& L_grad) {
        static const char* function =
          "stan::variational::advi.calc_combined_grad(%1%)";

        int dim       = muL.dimension();
        double tmp_lp = 0.0;

        stan::math::check_size_match(function,
                              "Dimension of mu grad vector", mu_grad.size(),
                              "Dimension of mean vector in variational q", dim);
        stan::math::check_square(function, "Scale matrix", L_grad);
        stan::math::check_size_match(function,
                              "Dimension of scale matrix", L_grad.rows(),
                              "Dimension of mean vector in variational q", dim);

        // Initialize everything to zero
        mu_grad = Eigen::VectorXd::Zero(dim);
        L_grad  = Eigen::MatrixXd::Zero(dim,dim);
        Eigen::VectorXd tmp_mu_grad = Eigen::VectorXd::Zero(dim);
        Eigen::VectorXd z_check = Eigen::VectorXd::Zero(dim);
        Eigen::VectorXd z_tilde = Eigen::VectorXd::Zero(dim);

        // Naive Monte Carlo integration
        for (int i = 0; i < n_monte_carlo_grad_; ++i) {

          // Draw from standard normal and transform to unconstrained space
          for (int d = 0; d < dim; ++d) {
            z_check(d) = stan::prob::normal_rng(0,1,rng_);
          }
          z_tilde = muL.to_unconstrained(z_check);

          // Compute gradient step in unconstrained space
          stan::model::gradient(model_, z_tilde, tmp_lp, tmp_mu_grad,
                                &std::cout);
          stan::agrad::recover_memory();

          // Update mu
          mu_grad += tmp_mu_grad;

          // Update L (lower triangular)
          for (int ii = 0; ii < dim; ++ii) {
            for (int jj = 0; jj <= ii; ++jj) {
              L_grad(ii,jj) += tmp_mu_grad(ii) * z_check(jj);
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
      void calc_combined_grad(
        advi_params_meanfield const& musigmatilde,
        Eigen::VectorXd& mu_grad,
        Eigen::VectorXd& sigma_tilde_grad) {
        static const char* function =
          "stan::variational::advi.calc_combined_grad(%1%)";

        int dim       = musigmatilde.dimension();
        double tmp_lp = 0.0;

        stan::math::check_size_match(function,
                              "Dimension of mu grad vector", mu_grad.size(),
                              "Dimension of mean vector in variational q", dim);

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
            z_check(d) = stan::prob::normal_rng(0,1,rng_);
          }
          z_tilde = musigmatilde.to_unconstrained(z_check);

          stan::math::check_not_nan(function, "z_tilde", z_tilde);

          // Compute gradient step in unconstrained space
          stan::model::gradient(model_, z_tilde, tmp_lp, tmp_mu_grad,
                                &std::cout);
          stan::agrad::recover_memory();

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
       * FULL-RANK ROBBINS-MONRO
       *
       * Runs Robbins-Monro Stochastic Gradient for some number of iterations
       *
       * @param muL            mean and cholesky factor of affine transform
       * @param tol_rel_obj    relative tolerance parameter for convergence
       * @param max_iterations max number of iterations to run algorithm
       */
      void do_robbins_monro_adagrad( advi_params_fullrank& muL,
                                     double tol_rel_obj,
                                     int max_iterations ) {
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
        int cb_size = (int)
                std::max(0.1*max_iterations/static_cast<double>(refresh_),1.0);
        boost::circular_buffer<double> cb(cb_size);

        // Compute difference between agrad::var version of log_prob
        // and double version. This is used in the ELBO calculation.
        int dim = muL.dimension();
        Eigen::VectorXd z_tmp = Eigen::VectorXd::Zero(dim);
        Eigen::Matrix<stan::agrad::var,Eigen::Dynamic,1> z_tmp_agrad(dim);
        for (int var_index = 0; var_index < dim; ++var_index) {
            z_tmp_agrad(var_index) = stan::agrad::var(0.0);
          }
        double constant_factor = (model_.template
                   log_prob<true,true>(z_tmp_agrad, &std::cout)).val()
                   -
                   (model_.template log_prob<false,true>(z_tmp, &std::cout));

        // Print stuff
        std::cout << "  iter"
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
                       + post_factor * mu_grad.array().square();
          L_s.array()  = pre_factor * L_s.array()
                       + post_factor * L_grad.array().square();

          // Take ADAgrad or rmsprop step
          muL.set_mu( muL.mu().array() +
            eta_stepsize_ * mu_grad.array() / (tau + mu_s.array().sqrt()) );
          muL.set_L_chol(  muL.L_chol().array()  +
            eta_stepsize_ * L_grad.array()  / (tau + L_s.array().sqrt()) );

          // Check for convergence every "refresh_"th iteration
          if (iter_counter % refresh_ == 0) {
            elbo_prev = elbo;
            elbo = calc_ELBO(muL, constant_factor);
            delta_elbo = rel_decrease(elbo, elbo_prev);
            cb.push_back(delta_elbo);
            delta_elbo_ave = std::accumulate(cb.begin(), cb.end(), 0.0)
                             / (double)(cb.size());
            delta_elbo_med = circ_buff_median(cb);
            std::cout << "  " << std::setw(4) << iter_counter
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
              std::cout << "   MEAN ELBO CONVERGED";
              do_more_iterations = false;
            }

            if (delta_elbo_med < tol_rel_obj) {
              std::cout << "   MEDIAN ELBO CONVERGED";
              do_more_iterations = false;
            }

            if (iter_counter > 100){
              if (delta_elbo_med > 0.5 || delta_elbo_ave > 0.5) {
                std::cout << "   MAY BE DIVERGING... INSPECT ELBO";
              }
            }

            std::cout << std::endl;
          }

          // Check for max iterations
          if (iter_counter == max_iterations) {
            std::cout << "MAX ITERATIONS" << std::endl;
            do_more_iterations = false;
          }

          ++iter_counter;
        }
      }


      /**
       * MEAN-FIELD ROBBINS-MONRO
       *
       * Runs Robbins-Monro Stochastic Gradient for some number of iterations
       *
       * @param musigmatilde    mean and log-std vector of affine transform
       * @param tol_rel_obj   relative tolerance parameter for convergence
       * @param max_iterations  max number of iterations to run algorithm
       */
      void do_robbins_monro_adagrad( advi_params_meanfield& musigmatilde,
                                     double tol_rel_obj,
                                     int max_iterations ) {

        // Gradients
        Eigen::VectorXd mu_grad           = Eigen::VectorXd::Zero(model_.num_params_r());
        Eigen::VectorXd sigma_tilde_grad  = Eigen::VectorXd::Zero(model_.num_params_r());

        // ADAgrad parameters
        double tau = 1.0;
        Eigen::VectorXd mu_s          = Eigen::VectorXd::Zero(model_.num_params_r());
        Eigen::VectorXd sigma_tilde_s = Eigen::VectorXd::Zero(model_.num_params_r());

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
        int cb_size = (int)
                std::max(0.1*max_iterations/static_cast<double>(refresh_),1.0);
        boost::circular_buffer<double> cb(cb_size);

        // Compute difference between agrad::var version of log_prob
        // and double version. This is used in the ELBO calculation.
        int dim = musigmatilde.dimension();
        Eigen::VectorXd z_tmp = Eigen::VectorXd::Zero(dim);
        Eigen::Matrix<stan::agrad::var,Eigen::Dynamic,1> z_tmp_agrad(dim);
        for (int var_index = 0; var_index < dim; ++var_index) {
            z_tmp_agrad(var_index) = stan::agrad::var(0.0);
          }
        double constant_factor = (model_.template
                   log_prob<true,true>(z_tmp_agrad, &std::cout)).val()
                   -
                   (model_.template log_prob<false,true>(z_tmp, &std::cout));

        // Print stuff
        std::cout << "  iter"
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
          sigma_tilde_s.array()  = pre_factor * sigma_tilde_s.array()
                                 + post_factor * sigma_tilde_grad.array().square();

          // Take ADAgrad or rmsprop step
          musigmatilde.set_mu(
            musigmatilde.mu().array() +
            eta_stepsize_ * mu_grad.array() / (tau + mu_s.array().sqrt())
            );
          musigmatilde.set_sigma_tilde(
            musigmatilde.sigma_tilde().array()  +
            eta_stepsize_ * sigma_tilde_grad.array()  / (tau + sigma_tilde_s.array().sqrt())
            );

          // Check for convergence every "refresh_"th iteration
          if (iter_counter % refresh_ == 0) {
            elbo_prev = elbo;
            elbo = calc_ELBO(musigmatilde, constant_factor);
            delta_elbo = rel_decrease(elbo, elbo_prev);
            cb.push_back(delta_elbo);
            delta_elbo_ave = std::accumulate(cb.begin(), cb.end(), 0.0)
                             / (double)(cb.size());
            delta_elbo_med = circ_buff_median(cb);
            std::cout << "  " << std::setw(4) << iter_counter
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
              std::cout << "   MEAN ELBO CONVERGED";
              do_more_iterations = false;
            }

            if (delta_elbo_med < tol_rel_obj) {
              std::cout << "   MEDIAN ELBO CONVERGED";
              do_more_iterations = false;
            }

            if (iter_counter > 100){
              if (delta_elbo_med > 0.5 || delta_elbo_ave > 0.5) {
                std::cout << "   MAY BE DIVERGING... INSPECT ELBO";
              }
            }

            std::cout << std::endl;
          }

          // Check for max iterations
          if (iter_counter == max_iterations) {
            std::cout << "MAX ITERATIONS REACHED" << std::endl;
            do_more_iterations = false;
          }

          ++iter_counter;
        }
      }

      void run_fullrank(double tol_rel_obj, int max_iterations) {
        std::cout
        << "This is stan::variational::advi::run_fullrank()" << std::endl;

        // Initialize variational parameters: mu, L
        Eigen::VectorXd mu = cont_params_;
        Eigen::MatrixXd L  = Eigen::MatrixXd::Identity(model_.num_params_r(),
                                                       model_.num_params_r());
        advi_params_fullrank muL = advi_params_fullrank(mu,L);

        // Robbins Monro ADAgrad
        do_robbins_monro_adagrad(muL, tol_rel_obj, max_iterations);

        cont_params_ = muL.mu();

        // std::cout
        // << "mu = " << std::endl
        // << muL.mu() << std::endl;

        // std::cout
        // << "Sigma = " << std::endl
        // << muL.L_chol() * muL.L_chol().transpose() << std::endl;

        // std::stringstream s;
        // stan::io::dump_writer writer(s);
        // writer.write("mu", muL.mu());
        // s << "\n";
        // writer.write("L_chol", muL.L_chol());
        // std::string written = s.str();
        // std::cout << written << std::endl;

        return;
      }

      void run_meanfield(double tol_rel_obj, int max_iterations) {
        std::cout
        << "This is stan::variational::advi::run_meanfield()" << std::endl;

        // Initialize variational parameters: mu, sigma_tilde
        Eigen::VectorXd mu           = cont_params_;
        Eigen::MatrixXd sigma_tilde  = Eigen::VectorXd::Constant(
                                                model_.num_params_r(),
                                                1.0);
        advi_params_meanfield musigmatilde = advi_params_meanfield(mu,sigma_tilde);

        // Robbins Monro ADAgrad
        do_robbins_monro_adagrad(musigmatilde, tol_rel_obj, max_iterations);

        cont_params_ = musigmatilde.mu();

        // std::cout
        // << "mu = " << std::endl
        // << musigmatilde.mu() << std::endl;

        // std::cout
        // << "sigma_tilde = " << std::endl
        // << musigmatilde.sigma_tilde() << std::endl;

        return;
      }

      Eigen::VectorXd const& cont_params() {
        return cont_params_;
      }

      // Compute the median of a circular buffer
      double circ_buff_median(boost::circular_buffer<double> const& cb) {

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

      // double rel_param_decrease(Eigen::VectorXd const& prev,
      //                           Eigen::VectorXd const& curr) const {
      //   return (prev - curr).norm() / std::max(prev.norm(),
      //                                          std::max(curr.norm(),
      //                                                   1.0));
      // }

      double rel_decrease(double prev, double curr) const {
        return std::abs(curr - prev) / std::abs(prev);
      }

    protected:

      M& model_;
      Eigen::VectorXd& cont_params_;
      double elbo_;
      BaseRNG& rng_;
      int n_monte_carlo_grad_;
      int n_monte_carlo_elbo_;
      double eta_stepsize_;
      int refresh_;
      std::ostream* out_stream_;
      std::ostream* diag_stream_;

    };

  } // variational

} // stan

#endif

