#ifndef __STAN__OPTIMIZATION__NEWTON_HPP__
#define __STAN__OPTIMIZATION__NEWTON_HPP__

#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <stan/model/prob_grad.hpp>

namespace stan {

  namespace optimization {

    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> matrix_d;
    typedef Eigen::Matrix<double, Eigen::Dynamic, 1> vector_d;

    namespace {
      // Negates any positive eigenvalues in H so that H is negative
      // definite, and then solves Hu = g and stores the result into
      // g. Avoids problems due to non-log-concave distributions.
      void make_negative_definite_and_solve(matrix_d& H, vector_d& g) {
        Eigen::SelfAdjointEigenSolver<matrix_d> solver(H);
        matrix_d eigenvectors = solver.eigenvectors();
        vector_d eigenvalues = solver.eigenvalues();
        vector_d eigenprojections = eigenvectors.transpose() * g;
        for (int i = 0; i < g.size(); i++) {
          eigenprojections[i] = -eigenprojections[i] / fabs(eigenvalues[i]);
        }
        g = eigenvectors * eigenprojections;
      }
    }

    double newton_step(stan::model::prob_grad& model, 
                       std::vector<double>& params_r,
                       std::vector<int>& params_i,
                       std::ostream* output_stream = 0) {
        std::vector<double> gradient;
        std::vector<double> hessian;
        
        double f0 = model.grad_hess_log_prob(params_r, params_i, 
                                             gradient, hessian);
        matrix_d H(params_r.size(), params_r.size());
        for (size_t i = 0; i < hessian.size(); i++) {
          H(i) = hessian[i];
        }
        vector_d g(params_r.size());
        for (size_t i = 0; i < gradient.size(); i++)
          g(i) = gradient[i];
        make_negative_definite_and_solve(H, g);
//         H.ldlt().solveInPlace(g);

        std::vector<double> new_params_r(params_r.size());
        double step_size = 2;
        double f1 = -1e100;
        while (!(f1 >= f0)) {
          step_size *= 0.5;
          for (size_t i = 0; i < params_r.size(); i++)
            new_params_r[i] = params_r[i] - step_size * g[i];
          try {
            f1 = model.grad_log_prob(new_params_r, params_i, gradient);
          } catch (std::exception& e) {
            f1 = -1e100;
          }
        }
        for (size_t i = 0; i < params_r.size(); i++)
          params_r[i] = new_params_r[i];

        return f1;
    }

  }

}

#endif
