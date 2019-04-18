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

    template<typename Model>
    class scaled_hessian_vector {
    private:
      const Model& model_;
      const Eigen::MatrixXd& L_;
      const Eigen::VectorXd& q_;
    public:
      scaled_hessian_vector(const Model& model,
			    const Eigen::MatrixXd& L,
			    const Eigen::VectorXd& q) : model_(model),
							L_(L),
							q_(q) {}

      int rows() { return q_.size(); }
      int cols() { return q_.size(); }

      void perform_op(const double* x_in, double* y_out) {
	Eigen::Map<const Eigen::VectorXd> x(x_in, cols());
	Eigen::Map<Eigen::VectorXd> y(y_out, rows());
	
	double lp;
	Eigen::VectorXd grad1;
	Eigen::VectorXd grad2;
	//stan::math::hessian_times_vector(log_prob_wrapper_covar<Model>(model), q, x, lp, Ax);
	double dx = 1e-5;
	Eigen::VectorXd dr = L_ * x * dx;
	stan::math::gradient(log_prob_wrapper_covar<Model>(model_), q_ + dr / 2.0, lp, grad1);
	stan::math::gradient(log_prob_wrapper_covar<Model>(model_), q_ - dr / 2.0, lp, grad2);
	y = L_.transpose() * (grad1 - grad2) / dx;
      }
    };

    class switching_adaptation: public windowed_adaptation {
    public:
      explicit switching_adaptation(int n)
        : windowed_adaptation("covariance") {}

      Eigen::MatrixXd covariance(const Eigen::MatrixXd& Y) {
	Eigen::MatrixXd centered = Y.colwise() - Y.rowwise().mean();
	return centered * centered.transpose() / std::max(centered.rows() - 1.0, 1.0);
      }
      
      template<typename Model>
      double top_eigenvalue(const Model& model, const Eigen::MatrixXd& L, const Eigen::VectorXd& q) {
	Eigen::VectorXd eigenvalues;
	Eigen::MatrixXd eigenvectors;

	scaled_hessian_vector<Model> op(model, L, q);

	Spectra::SymEigsSolver<double, Spectra::LARGEST_MAGN, decltype(op)> eigs(&op, 1, 2);
	eigs.init();
	eigs.compute();

	if(eigs.info() != Spectra::SUCCESSFUL) {
	  throw std::domain_error("Failed to compute eigenvalue of Hessian of log density. The switching metric requires these");
	}

	return eigs.eigenvalues()(0);
      }

      double bottom_eigenvalue_estimate(const Eigen::MatrixXd& L, const Eigen::MatrixXd& covar) {
	Eigen::MatrixXd S = L.template triangularView<Eigen::Lower>().
	  solve(L.template triangularView<Eigen::Lower>().solve(covar).transpose()).transpose();

	Spectra::DenseSymMatProd<double> op(S);
	Spectra::SymEigsSolver<double, Spectra::LARGEST_MAGN, decltype(op)> eigs(&op, 1, 2);
	eigs.init();
	eigs.compute();

	if(eigs.info() != Spectra::SUCCESSFUL) {
	  throw std::domain_error("Failed to compute eigenvalue of covariance of log density. The switching metric requires these");
	}

	return -1.0 / eigs.eigenvalues()(0);
      }

      template<typename Model>
      bool learn_covariance(const Model& model, Eigen::MatrixXd& covar, const Eigen::VectorXd& q) {
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
	      
	      std::cout << "train: " << Y.cols() - Ntest << ", test: " << Ntest << std::endl;
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

	      double low_eigenvalue_dense = bottom_eigenvalue_estimate(L_dense, cov_test);
	      double low_eigenvalue_diag = bottom_eigenvalue_estimate(L_diag, cov_test);

	      double c_dense = 0.0;
	      double c_diag = 0.0;
	      for(int i = 0; i < 5; i++) {
		double high_eigenvalue_dense = top_eigenvalue(model, L_dense, Ytest.block(0, i, N, 1));
		double high_eigenvalue_diag = top_eigenvalue(model, L_diag, Ytest.block(0, i, N, 1));

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
	      } else {
		covar = diag;
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
