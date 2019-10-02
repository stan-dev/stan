#ifndef STAN_SERVICES_KHAT_HPP
#define STAN_SERVICES_KHAT_HPP

#include <stan/math/prim/mat/fun/mean.hpp>
#include <stan/math/rev/scal/fun/exp.hpp>
#include <stan/services/khat.hpp>
#include <Eigen/Dense>
#include <type_traits>
#include <vector>

namespace stan {
namespace math {
namespace services {
namespace experimental {
namespace advi {

  template <typename T_x>
  int lx(std::vector<T_x>& a, const Eigen::Matrix<T_x, -1, 1>& x) {
    size_t a_size = a.size();
    size_t x_rows = x.rows();

    std::vector<T_x> a_temp(a_size);
    for (size_t i = 0; i < a_size; ++i)
      a_temp[i] = -a[i];

    std::vector<T_x> k(a_size);
    std::vector<T_x> temp_vec(x_rows);

    for (size_t i = 0; i < a_size; ++i) {
      for (size_t ii = 0; ii < x_rows; ++ii) {
        temp_vec[ii] = a_temp[i] * x[ii];
      } 
      for (size_t ii = 0; ii < x_rows; ++ii) {
        temp_vec[ii] = std::log1p(temp_vec[ii]);
      }
      k[i] = mean(temp_vec);
    }

    for (size_t i = 0; i < a_size; ++i) {
      a[i] = std::log(a_temp[i] / k[i]) - k[i] - 1;
    }
    return 0;
  }

  template <typename T_k>
  int adjust_k_wip(T_k& k, const size_t& n) {
    size_t a = 10;
    size_t n_plus_a = n + a;
    k = k * n / n_plus_a + a * 0.5 / n_plus_a;
    return 0;
  }

  template <typename T_x>
  int compute_khat(const std::vector<T_x>& x,
                   const int& min_grid_points = 30,
                   T_x &k = std::numeric_limits<double>::quiet_NaN()) {
    size_t N = x.size();
    std::vector<T_x> x_s(N);
    x_s = x;
    std::sort(x_s.begin(), x_s.end());

    double prior = 3.0;
    double root_N = std::sqrt(N);
    int M = min_grid_points + std::floor(root_N);

    std::vector<int> jj(M);
    for (size_t i = 1; i <= M; ++i)
      jj[i] = i;
    size_t quart1 = std::floor(N / 4.0 + 0.5);
    auto x_star = x_s[quart1 - 1];

    std::vector<double> theta(M);
    for(size_t i = 0; i < M; ++i) {
      theta[i] = std::exp(std::log(1.0) - std::log(x_s[N - 1]))
        + (1.0 - std::sqrt(std::exp(std::log(M) - std::log(jj[i + 1] - .5)))) /
      prior / x_star;
    }

    Eigen::Matrix<double, -1, 1> x_lx;
    x_lx.resize(N, 1);
    for (size_t i = 0; i < N; ++i)
      x_lx[i] = x[i];

    std::vector<T_x> l_theta(M);
    for (int i = 0; i < M; ++i)
      l_theta[i] = theta[i];
    lx(l_theta, x_lx);
    for (int i = 0; i < M; ++i)
      l_theta[i] = N * l_theta[i];

    std::vector<T_x> w_theta(M);
    for (size_t i = 0; i < M; ++i) {
      w_theta[i] = 0;
      for (size_t ii = 0; ii < M; ++ii)
        w_theta[i] = w_theta[i] + std::exp(l_theta[ii] - l_theta[i]);
      w_theta[i] = 1 / w_theta[i];
    }    
    auto theta_hat = 0.0;
    for (size_t i = 0; i < M; ++i) {
      theta_hat = theta_hat + theta[i] * w_theta[i];
    }

    std::vector<double> k_vec(N);
    for (size_t i = 0; i < N; ++i) {
      k_vec[i] = log1p(-theta_hat * x_s[i]);
    }

    k = mean(k_vec);
    auto sigma = -k / theta_hat;
    adjust_k_wip(k, N);

    if (std::numeric_limits<double>::quiet_NaN() == k)
      k = std::numeric_limits<double>::infinity();

    return 0;
  }
}  // namespace advi
}  // namespace experimental
}  // namespace services
}  // namespace math
}  // namespace stan
#endif
