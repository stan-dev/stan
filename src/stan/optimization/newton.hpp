#ifndef STAN_OPTIMIZATION_NEWTON_HPP
#define STAN_OPTIMIZATION_NEWTON_HPP

#include <stan/model/grad_hess_log_prob.hpp>
#include <stan/model/log_prob_grad.hpp>
#include <stan/math/prim/fun/Eigen.hpp>
#include <cmath>
#include <limits>
#include <vector>

namespace stan {
namespace optimization {

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> matrix_d;
typedef Eigen::Matrix<double, Eigen::Dynamic, 1> vector_d;

/**
 * Negates any positive eigenvalues in H so that H is negative
 * definite, then solves Hu = g and stores the result into g.
 * Avoids problems due to non-log-concave distributions.
 *
 * Each eigenvalue magnitude is floored at delta before inverting, so the
 * step along a direction with little or no curvature is a gradient step
 * scaled by 1 / delta rather than an unbounded or undefined quantity.
 * This "saturating inverse" is continuous in the eigenvalues and bounds
 * the effective condition number of the solve by 1 / sqrt(u).
 *
 * The floor is delta = max(sqrt(u) * max|lambda|, sqrt(u)), with u the
 * unit roundoff. The relative term follows Nocedal and Wright, Numerical
 * Optimization, 2nd ed., Section 3.4, which replaces problem eigenvalues
 * with a delta of order sqrt(u). The absolute term is the same value
 * under a well-scaled assumption and keeps an all-zero Hessian well
 * defined. The backtracking line search in newton_step shortens any
 * step that turns out too long.
 *
 * @param[in] H Hessian of the log density
 * @param[in, out] g gradient on input, Newton step direction on output
 */
inline void make_negative_definite_and_solve(matrix_d& H, vector_d& g) {
  Eigen::SelfAdjointEigenSolver<matrix_d> solver(H);
  matrix_d eigenvectors = solver.eigenvectors();
  vector_d eigenvalues = solver.eigenvalues();
  vector_d eigenprojections = eigenvectors.transpose() * g;
  const double sqrt_eps = std::sqrt(std::numeric_limits<double>::epsilon());
  double max_abs_eigenvalue = eigenvalues.cwiseAbs().maxCoeff();
  double delta = std::fmax(sqrt_eps * max_abs_eigenvalue, sqrt_eps);
  for (int i = 0; i < g.size(); i++) {
    eigenprojections[i]
        = -eigenprojections[i] / std::fmax(std::fabs(eigenvalues[i]), delta);
  }
  g = eigenvectors * eigenprojections;
}

/**
 * Returns true if every element of the vector is finite.
 *
 * @tparam Vec vector type with size() and operator[]
 * @param[in] v vector to check
 * @return true if all elements are finite
 */
template <typename Vec>
inline bool all_finite(const Vec& v) {
  for (int i = 0; i < static_cast<int>(v.size()); ++i) {
    if (!std::isfinite(v[i])) {
      return false;
    }
  }
  return true;
}

template <typename M, bool jacobian = false>
double newton_step(M& model, std::vector<double>& params_r,
                   std::vector<int>& params_i,
                   std::ostream* output_stream = 0) {
  std::vector<double> gradient;
  std::vector<double> hessian;

  double f0 = stan::model::grad_hess_log_prob<true, jacobian>(
      model, params_r, params_i, gradient, hessian);
  if (!std::isfinite(f0)) {
    return f0;
  }
  matrix_d H(params_r.size(), params_r.size());
  for (size_t i = 0; i < hessian.size(); i++) {
    H(i) = hessian[i];
  }
  vector_d g(params_r.size());
  for (size_t i = 0; i < gradient.size(); i++)
    g(i) = gradient[i];
  make_negative_definite_and_solve(H, g);
  if (!all_finite(g)) {
    return f0;
  }

  std::vector<double> new_params_r(params_r.size());
  double step_size = 2;
  double min_step_size = 1e-50;
  double f1 = -1e100;

  while (f1 < f0) {
    step_size *= 0.5;
    if (step_size < min_step_size)
      return f0;

    for (size_t i = 0; i < params_r.size(); i++)
      new_params_r[i] = params_r[i] - step_size * g[i];
    if (!all_finite(new_params_r)) {
      f1 = -1e100;
      continue;
    }
    try {
      f1 = stan::model::log_prob_grad<true, jacobian>(model, new_params_r,
                                                      params_i, gradient);
    } catch (std::domain_error& e) {
      // FIXME:  this is not a good way to handle a general exception
      f1 = -1e100;
    }
    if (!std::isfinite(f1)) {
      f1 = -1e100;
    }
  }
  for (size_t i = 0; i < params_r.size(); i++)
    params_r[i] = new_params_r[i];

  return f1;
}

}  // namespace optimization
}  // namespace stan
#endif
