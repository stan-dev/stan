#ifndef STAN_SERVICES_EXPERIMENTAL_ADVI_KHAT_HPP
#define STAN_SERVICES_EXPERIMENTAL_ADVI_KHAT_HPP

#include <stan/math/rev/scal/fun/floor.hpp>
#include <stan/math/rev/scal/fun/sqrt.hpp>
#include <stan/math/rev/scal/fun/log.hpp>
#include <Eigen/Dense>

namespace stan {
namespace math {
namespace services {
namespace experimental {
namespace advi {

  <typename T_x>
  int compute_khat(const std::vector<T_x>& x) {
    return 0;
  }

  <typename T_x>
  int lx(std::vector<T_x>& a, const Eigen::Matrix<T_x, -1, 1>& x) {
    size_t a_size = a.size();
    size_t x_rows = x.rows();

    std::vector<T_x> a_temp(a_size);
    for (size_t i = 0; i < a_size; ++i)
      a_temp[i] = -a[i];

    std::vector<T_x> k(a_size);
    for (size_t i = 0; i < a_size; ++i)
      k[i] = mean(log(a_temp[i] * x));

    for (size_t i = 0; i < a_size; ++i)
      a[i] = log(a_temp[i] / k[i]) - k[i] - 1;

    return 0;
  }

  typename<T_k>
  int adjust_k_wip(T_k& k, const size_t& n) {
    size_t a = 10;
    size_t n_plus_a = n + a;
    k = k * n / n_plus_a + a * 0.5 / n_plus_a;
    return 0;
  }
  

}  // namespace advi
}  // namespace experimental
}  // namespace services
}  // namespace math
}  // namespace stan
