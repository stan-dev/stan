#ifndef STAN_MCMC_SWITCHING_ADAPTATION_HPP
#define STAN_MCMC_SWITCHING_ADAPTATION_HPP

#include <stan/math/prim/mat.hpp>
#include <stan/mcmc/windowed_adaptation.hpp>
#include <vector>

namespace stan {

  namespace mcmc {
    template <typename Model>
    struct log_prob_wrapper_covar {
      const Model& model_;
      log_prob_wrapper_covar(const Model& model) : model_(model) {}
      
      template <typename T>
      T operator()(const Eigen::Matrix<T, Eigen::Dynamic, 1>& q) const {
	return model_.template log_prob<true, true, T>(const_cast<Eigen::Matrix<T, Eigen::Dynamic, 1>& >(q), &std::cout);
      }
    };

    class switching_adaptation: public windowed_adaptation {
    public:
      explicit switching_adaptation(int n)
        : windowed_adaptation("covariance") {}

      /**
       * Compute the covariance of data in Y. Rows of Y are different data. Columns of Y are different variables.
       *
       * @param Y Data
       * @return Covariance of Y
       */
      Eigen::MatrixXd covariance(const Eigen::MatrixXd& Y) {
	Eigen::MatrixXd centered = Y.colwise() - Y.rowwise().mean();
	return centered * centered.transpose() / std::max(centered.rows() - 1.0, 1.0);
      }

      /**
       * Compute the largest magnitude eigenvalue of a symmetric matrix using the power method. The function f
       *  should return the product of that matrix with an abitrary vector.
       *
       * f should take one Eigen::VectorXd argument, x, and return the product of a matrix with x as
       *  an Eigen::VectorXd argument of the same size.
       *
       * The eigenvalue is estimated iteratively. If the kth estimate is e_k, then the function returns when
       *  either abs(e_{k + 1} - e_k) < tol * abs(e_k) or the maximum number of iterations have been performed
       *
       * This means the returned eigenvalue might not be computed to full precision
       *
       * @param initial_guess Initial guess of the eigenvector of the largest eigenvalue
       * @param max_iterations Maximum number of power iterations
       * @param tol Relative tolerance
       * @return Largest magnitude eigenvalue of operator f
       */
      template<typename F>
      double power_method(F& f, const Eigen::VectorXd& initial_guess, int max_iterations, double tol) {
	Eigen::VectorXd v = initial_guess;
	double eval = 0.0;

	for(int i = 0; i < max_iterations; i++) {
	  Eigen::VectorXd Av = f(v);
	  double v_norm = v.norm();
	  double new_eval = v.dot(Av) / (v_norm * v_norm);
	  if(std::abs(new_eval - eval) <= tol * std::abs(eval)) {
	    std::cout << "Converged at i = " << i << std::endl;
	    eval = new_eval;
	    break;
	  }
	  eval = new_eval;
	  v = Av / Av.norm();
	}

	return eval;
      }

      /**
       * Compute the largest eigenvalue of the Hessian of the log density rescaled by a metric,
       *  that is, the largest eigenvalue of L^T \nabla^2_{qq} H(q) L
       *
       * @tparam Model Type of model
       * @param model Defines the log density
       * @param q Point around which to compute the Hessian
       * @param L Cholesky decomposition of Metric
       * @return Largest eigenvalue
       */
      template<typename Model>
      double eigenvalue_scaled_hessian(const Model& model, const Eigen::MatrixXd& L, const Eigen::VectorXd& q) {
	Eigen::VectorXd eigenvalues;
	Eigen::MatrixXd eigenvectors;

	auto hessian_vector = [&](const Eigen::VectorXd& x) -> Eigen::VectorXd {
	  double lp;
	  Eigen::VectorXd grad1;
	  Eigen::VectorXd grad2;
	  //stan::math::hessian_times_vector(log_prob_wrapper_covar<Model>(model), q, x, lp, Ax);
	  double dx = 1e-5;
	  Eigen::VectorXd dr = L * x * dx;
	  stan::math::gradient(log_prob_wrapper_covar<Model>(model), q + dr / 2.0, lp, grad1);
	  stan::math::gradient(log_prob_wrapper_covar<Model>(model), q - dr / 2.0, lp, grad2);
	  return L.transpose() * (grad1 - grad2) / dx;
	};

	return power_method(hessian_vector, Eigen::VectorXd::Random(q.size()), 100, 1e-3);
      }

      /**
       * Compute the largest eigenvalue of the sample covariance rescaled by a metric,
       *  that is, the largest eigenvalue of L^{-T} \Sigma L^{-1}
       *
       * @param L Cholesky decomposition of Metric
       * @param Sigma Sample covariance
       * @return Largest eigenvalue
       */
      double eigenvalue_scaled_covariance(const Eigen::MatrixXd& L, const Eigen::MatrixXd& Sigma) {
	Eigen::MatrixXd S = L.template triangularView<Eigen::Lower>().
	  solve(L.template triangularView<Eigen::Lower>().solve(Sigma).transpose()).transpose();

	auto Sx = [&](Eigen::VectorXd x) -> Eigen::VectorXd {
	  return S * x;
	};
	
	return power_method(Sx, Eigen::VectorXd::Random(Sigma.cols()), 100, 1e-3);
      }

      /**
       * Update the metric if at the end of an adaptation window.
       *
       * @tparam Model Type of model
       * @param model Defines the log density
       * @param covar[out] New metric
       * @param covar_is_diagonal[out] Set to true if metric is diagonal, false otherwise
       * @param q New MCMC draw
       * @return True if this was the end of an adaptation window, false otherwise
       */
      template<typename Model>
      bool learn_covariance(const Model& model, Eigen::MatrixXd& covar, bool& covar_is_diagonal, const Eigen::VectorXd& q) {
        if (adaptation_window())
	  qs_.push_back(q);

        if (end_adaptation_window()) {
          compute_next_window();

	  int N = q.size();
	  int M = qs_.size();

	  Eigen::MatrixXd Y = Eigen::MatrixXd::Zero(N, M);
	  std::vector<int> idxs(M);
	  for(int i = 0; i < qs_.size(); i++)
	    idxs[i] = i;

	  std::random_shuffle(idxs.begin(), idxs.end());
	  for(int i = 0; i < qs_.size(); i++)
	    Y.block(0, i, N, 1) = qs_[idxs[i]];

	  bool use_dense = false;
	  for(auto state : { "selection", "refinement" }) {
	    Eigen::MatrixXd Ytrain;
	    Eigen::MatrixXd Ytest;

	    if(state == "selection") {
	      int Ntest;
	      Ntest = int(0.2 * Y.cols());
	      if(Ntest < 5) {
		Ntest = 5;
	      }

	      if(Y.cols() < 10) {
		throw std::runtime_error("Each warmup stage must have at least 10 samples");
	      }
	      
	      Ytrain = Y.block(0, 0, N, Y.cols() - Ntest);
	      Ytest = Y.block(0, Ytrain.cols(), N, Ntest);
	    } else {
	      Ytrain = Y;
	    }

	    Eigen::MatrixXd cov_train = covariance(Ytrain);
	    Eigen::MatrixXd cov_test = covariance(Ytest);

	    Eigen::MatrixXd dense = (N / (N + 5.0)) * cov_train +
	      1e-3 * (5.0 / (N + 5.0)) * Eigen::MatrixXd::Identity(cov_train.rows(), cov_train.cols());
	    Eigen::MatrixXd diag = dense.diagonal().asDiagonal();

	    covar = dense;

	    if(state == "selection") {
	      Eigen::MatrixXd L_dense = dense.llt().matrixL();
	      Eigen::MatrixXd L_diag = diag.diagonal().array().sqrt().matrix().asDiagonal();

	      double low_eigenvalue_dense = -1.0 / eigenvalue_scaled_covariance(L_dense, cov_test);
	      double low_eigenvalue_diag = -1.0 / eigenvalue_scaled_covariance(L_diag, cov_test);

	      double c_dense = 0.0;
	      double c_diag = 0.0;
	      for(int i = 0; i < 5; i++) {
		double high_eigenvalue_dense = eigenvalue_scaled_hessian(model, L_dense, Ytest.block(0, i, N, 1));
		double high_eigenvalue_diag = eigenvalue_scaled_hessian(model, L_diag, Ytest.block(0, i, N, 1));

		c_dense = std::max(c_dense, std::sqrt(high_eigenvalue_dense / low_eigenvalue_dense));
		c_diag = std::max(c_diag, std::sqrt(high_eigenvalue_diag / low_eigenvalue_diag));
	      }

	      std::cout << "adapt: " << adapt_window_counter_ << ", which: dense, max: " << c_dense << std::endl;
	      std::cout << "adapt: " << adapt_window_counter_ << ", which: diag, max: " << c_diag << std::endl;

	      if(c_dense < c_diag) {
		use_dense = true;
	      } else {
		use_dense = false;
	      }
	    } else {
	      if(use_dense) {
		covar = dense;
		covar_is_diagonal = false;
	      } else {
		covar = diag;
		covar_is_diagonal = true;
	      }
	    }
	  }

          ++adapt_window_counter_;
	  qs_.clear();

          return true;
        }

        ++adapt_window_counter_;
        return false;
      }

    protected:
      std::vector< Eigen::VectorXd > qs_;
    };

  }  // mcmc

}  // stan

#endif
