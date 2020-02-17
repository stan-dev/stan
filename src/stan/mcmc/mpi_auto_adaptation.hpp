#ifndef STAN_MCMC_MPI_AUTO_ADAPTATION_HPP
#define STAN_MCMC_MPI_AUTO_ADAPTATION_HPP

#include <stan/math/prim/mat.hpp>
#include <stan/mcmc/mpi_metric_adaptation.hpp>
#include <vector>

#ifdef STAN_LANG_MPI
#include <stan/math/mpi/mpi_covar_estimator.hpp>
#endif

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
  
  namespace internal {
    /**
     * Compute the covariance of data in Y.
     *
     * Columns of Y are different variables. Rows are different samples.
     *
     * When there is only one row in Y, return a covariance matrix of the expected
     *  size filled with zeros.
     *
     * @param Y Data
     * @return Covariance of Y
     */
    Eigen::MatrixXd covariance(const Eigen::MatrixXd& Y) {
      stan::math::check_nonzero_size("covariance", "Y", Y);
	
      Eigen::MatrixXd centered = Y.rowwise() - Y.colwise().mean();
      return centered.transpose() * centered / std::max(centered.rows() - 1.0, 1.0);
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
     * @param[in,out] max_iterations Maximum number of power iterations, on return number of iterations used
     * @param[in,out] tol Relative tolerance, on return the relative error in the eigenvalue estimate
     * @return Largest magnitude eigenvalue of operator f
     */
    template<typename F>
    double power_method(F& f, const Eigen::VectorXd& initial_guess, int& max_iterations, double& tol) {
      Eigen::VectorXd v = initial_guess;
      double eval = 0.0;
      Eigen::VectorXd Av = f(v);
      stan::math::check_matching_sizes("power_method", "matrix vector product", Av, "vector", v);

      int i = 0;
      for(; i < max_iterations; ++i) {
	double v_norm = v.norm();
	double new_eval = v.dot(Av) / (v_norm * v_norm);
	if(i == max_iterations - 1 || std::abs(new_eval - eval) <= tol * std::abs(eval)) {
	  tol = std::abs(new_eval - eval) / std::abs(eval);
	  eval = new_eval;
	  max_iterations = i + 1;
	  break;
	}

	eval = new_eval;
	v = Av / Av.norm();

	Av = f(v);
      }

      return eval;
    }

    /**
     * Compute the largest eigenvalue of the sample covariance rescaled by a metric,
     *  that is, the largest eigenvalue of L^{-1} \Sigma L^{-T}
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

      int max_iterations = 100;
      double tol = 1e-3;

      return internal::power_method(Sx, Eigen::VectorXd::Random(Sigma.cols()), max_iterations, tol);
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
 
      int max_iterations = 100;
      double tol = 1e-3;
	
      return internal::power_method(hessian_vector, Eigen::VectorXd::Random(q.size()), max_iterations, tol);
    }
  }

template <typename Model>
class mpi_auto_adaptation : public mpi_metric_adaptation {
#ifdef STAN_LANG_MPI
  using est_t = stan::math::mpi::mpi_covar_estimator;

  int window_size_;
  int n_params_;
  Model& model_;
  std::deque<Eigen::VectorXd> last_qs_;
public:
  est_t estimator;
  bool is_diagonal_;

  mpi_auto_adaptation(Model& model, int n_params, int num_iterations, int window_size)
    : window_size_(window_size),
      n_params_(n_params),
      model_(model),
      estimator(n_params, num_iterations),
      is_diagonal_(false) {}

  virtual void add_sample(const Eigen::VectorXd& q, int curr_win_count) {
    estimator.add_sample(q);
    last_qs_.push_back(q);
    if(last_qs_.size() > 5) {
      last_qs_.pop_front();
    }
  }
    
  virtual void learn_metric(Eigen::MatrixXd& covar, int win, int curr_win_count,
			    const stan::math::mpi::Communicator& comm) {
    if(curr_win_count > 1) {
      win = std::max(1, win);
    }

    int col_begin = win * window_size_;
    int num_draws = (curr_win_count - win) * window_size_;
    
    int M = n_params_;

    try {
      bool use_dense = false;
      for(auto state : { "selection", "refinement" }) {
	Eigen::MatrixXd cov_train = Eigen::MatrixXd::Zero(M, M);
	Eigen::MatrixXd cov_test = Eigen::MatrixXd::Zero(M, M);

	int Ntest;
	if(state == "selection") {
	  Ntest = int(0.2 * num_draws);
	  if(Ntest < 5) {
	    Ntest = 5;
	  }

	  if(num_draws < 10) {
	    throw std::runtime_error("Each warmup stage must have at least 10 samples");
	  }

	  learn_covariance(cov_train, col_begin, num_draws - Ntest, comm);
	  learn_covariance(cov_test, col_begin + num_draws - Ntest, Ntest, comm);
	  //Ytrain = Y.block(0, 0, M, Y.cols() - Mtest);
	  //Ytest = Y.block(0, Ytrain.cols(), M, Mtest);
	} else {
	  learn_covariance(cov_train, col_begin, num_draws, comm);
	  Ntest = 0;
	  //Ytrain = Y;
	}
	
	Eigen::MatrixXd dense = ((num_draws - Ntest) / ((num_draws - Ntest) + 5.0)) * cov_train +
	  1e-3 * (5.0 / ((num_draws - Ntest) + 5.0)) * Eigen::MatrixXd::Identity(cov_train.rows(), cov_train.cols());

	Eigen::MatrixXd diag = dense.diagonal().asDiagonal();

	covar = dense;

	if(state == "selection") {
	  Eigen::MatrixXd L_dense = dense.llt().matrixL();
	  Eigen::MatrixXd L_diag = diag.diagonal().array().sqrt().matrix().asDiagonal();

	  double low_eigenvalue_dense = -1.0 / internal::eigenvalue_scaled_covariance(L_dense, cov_test);
	  double low_eigenvalue_diag = -1.0 / internal::eigenvalue_scaled_covariance(L_diag, cov_test);

	  std::cout << "TRAIN low:" << low_eigenvalue_dense << std::endl;
	  std::cout << "TEST low :" << low_eigenvalue_diag << std::endl;

	  double c_dense = 0.0;
	  double c_diag = 0.0;
	  for(int i = 0; i < last_qs_.size(); i++) {
	    double high_eigenvalue_dense = internal::eigenvalue_scaled_hessian(model_, L_dense, last_qs_[i]);
	    double high_eigenvalue_diag = internal::eigenvalue_scaled_hessian(model_, L_diag, last_qs_[i]);

	    c_dense = std::max(c_dense, std::sqrt(high_eigenvalue_dense / low_eigenvalue_dense));
	    c_diag = std::max(c_diag, std::sqrt(high_eigenvalue_diag / low_eigenvalue_diag));
	  }

	  std::cout << "adapt dense, max: " << c_dense << std::endl;
	  std::cout << "adapt diag, max: " << c_diag << std::endl;
	  
	  if(c_dense < c_diag) {
	    use_dense = true;
	  } else {
	    use_dense = false;
	  }
	} else {
	  if(use_dense) {
	    covar = dense;
	    is_diagonal_ = false;
	  } else {
	    covar = diag;
	    is_diagonal_ = true;
	  }
	}
      }
    } catch(const std::exception& e) {
      std::cout << e.what() << std::endl;
      std::cout << "Exception while using auto adaptation, falling back to diagonal" << std::endl;
      Eigen::MatrixXd cov = Eigen::MatrixXd::Zero(M, M);
      learn_covariance(cov, col_begin, num_draws, comm);
      covar = ((num_draws / (num_draws + 5.0)) * cov.diagonal()
	       + 1e-3 * (5.0 / (num_draws + 5.0)) * Eigen::VectorXd::Ones(cov.cols())).asDiagonal();
      is_diagonal_ = true;
    }

    std::cout << covar << std::endl;
  }

  void learn_covariance(Eigen::MatrixXd& covar,
			int col_begin, int n_samples,
			const stan::math::mpi::Communicator& comm) {
    estimator.sample_covariance(covar, col_begin, n_samples, comm);
    //double n = static_cast<double>(estimator.num_samples(comm));
    //covar = (n / (n + 5.0)) * covar
    //  + 1e-3 * (5.0 / (n + 5.0))
    //  * Eigen::MatrixXd::Identity(covar.rows(), covar.cols());
    // restart();
  }

  virtual void restart() {
    estimator.restart();
  }
#else
public:
  mpi_auto_adaptation(int n_params, int num_iterations, int window_size) {}
#endif
};

}  // namespace mcmc
}  // namespace stan

#endif
