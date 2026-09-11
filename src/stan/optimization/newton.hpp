#ifndef STAN_OPTIMIZATION_NEWTON_HPP
#define STAN_OPTIMIZATION_NEWTON_HPP

#include <stan/math/rev.hpp>
#include <stan/math/rev/functor/finite_diff_hessian_auto.hpp>
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
template <typename VecG>
inline void make_negative_definite_and_solve(matrix_d& H, VecG& g) {
  Eigen::SelfAdjointEigenSolver<matrix_d> solver(H);
  auto&& eigenvectors = solver.eigenvectors();
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
 * Take one Newton step on the log density of the model, updating
 * params_r in place.
 *
 * The gradient is computed by reverse-mode autodiff and the Hessian
 * by central finite differences of the gradient with a per-coordinate
 * step size, which costs 2 * params_r.size() + 1 gradient evaluations.
 * The Hessian is made negative definite before solving for the Newton
 * direction, and a backtracking line search on the log density chooses
 * the step length. The log density is evaluated with all constant
 * terms included (propto = false) so that the line search can use
 * plain double arithmetic without autodiff.
 *
 * @tparam M Class of model.
 * @tparam jacobian True if the log absolute Jacobian determinant of
 * the inverse parameter transforms is added to the log density.
 * @param[in] model Model.
 * @param[in, out] params_r Unconstrained parameters; updated to the
 * new point if the step improves the log density.
 * @param[in] params_i Integer-valued parameters (unused).
 * @param[in, out] output_stream Stream to which print statements in
 * the Stan program are written.
 * @return Log density, including constant terms, at the returned
 * params_r.
 */
template <typename M, bool jacobian = false>
double newton_step(M& model, std::vector<double>& params_r,
                   std::vector<int>& params_i,
                   std::ostream* output_stream = 0) {
  const Eigen::Index n = params_r.size();
  const vector_d x = Eigen::Map<const vector_d>(params_r.data(), n);

  auto log_density = [&](auto&& theta) {
    return model.template log_prob<false, jacobian, stan::math::var>(
        theta, output_stream);
  };
  double f0;
  vector_d g;
  matrix_d H;
  stan::math::internal::finite_diff_hessian_auto(log_density, x, f0, g, H);
  if (!std::isfinite(f0)) {
    return f0;
  }
  make_negative_definite_and_solve(H, g);
  if (!g.array().allFinite()) {
    return f0;
  }

  vector_d new_params_r(n);
  double step_size = 2;
  double min_step_size = 1e-50;
  double f1 = -1e100;

  while (f1 < f0) {
    step_size *= 0.5;
    if (step_size < min_step_size)
      return f0;

    new_params_r = x - step_size * g;
    if (!new_params_r.array().allFinite()) {
      f1 = -1e100;
      continue;
    }
    try {
      f1 = model.template log_prob<false, jacobian, double>(new_params_r,
                                                            output_stream);
    } catch (std::domain_error& e) {
      // FIXME:  this is not a good way to handle a general exception
      f1 = -1e100;
    }
    if (!std::isfinite(f1)) {
      f1 = -1e100;
    }
  }
  Eigen::Map<vector_d>(params_r.data(), n) = new_params_r;

  return f1;
}

}  // namespace optimization
}  // namespace stan
#endif
