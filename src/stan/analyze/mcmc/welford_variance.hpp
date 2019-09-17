#ifndef STAN_ANALYZE_MCMC_WELFORD_VARIANCE_HPP
#define STAN_ANALYZE_MCMC_WELFORD_VARIANCE_HPP

#include <stan/math/prim/mat.hpp>

namespace stan {
namespace analyze {

/**
 * Computes variance estimate using Welford's online algorithm; see
 * https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance.
 *
 * @param y draws from a single chain
 * @param ddof denominator degrees of freedom, defaults to 1
 * @return variance estimate
 */
template<typename Derived>
inline double welford_variance(const Eigen::MatrixBase<Derived>& y,
                               int ddof = 1) {
  double d = 0;
  double m = 0;
  double S = 0;
  int N = y.size();
  for (int n = 0; n < N; ++n) {
    d = y[n] - m;
    m += d / (n + 1);
    S += d * (y[n] - m);
  }
  return S / (N - ddof);
}

}  // namespace analyze
}  // namespace stan

#endif
