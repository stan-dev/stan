#ifndef STAN_SERVICES_PSIS_HPP
#define STAN_SERVICES_PSIS_HPP

#include <stan/math.hpp>
#include <stan/callbacks/interrupt.hpp>
#include <stan/callbacks/logger.hpp>
#include <stan/callbacks/writer.hpp>
#include <stan/io/var_context.hpp>
#include <stan/optimization/bfgs.hpp>
#include <stan/optimization/lbfgs_update.hpp>
#include <stan/services/error_codes.hpp>
#include <stan/services/util/initialize.hpp>
#include <stan/services/util/create_rng.hpp>
#include <tbb/parallel_for.h>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <mutex>
#include <queue>

namespace stan {
namespace services {
namespace psis {

inline Eigen::Array<double, -1, 1> lx(const Eigen::Array<double, -1, 1>& a,
                                      const Eigen::Array<double, -1, 1>& x) {
  double k = ((-a) * x).log1p().mean();
  return (-a / k).log() - k - 1;
}

auto gpdfit(const Eigen::Array<double, -1, 1>& x, bool wip = true,
            size_t min_grid_pts = 30) {
  // See section 4 of Zhang and Stephens (2009)
  const auto N = x.size();
  constexpr auto prior = 3.0;
  const auto M = min_grid_pts + std::floor(std::sqrt(N));
  Array<double, -1, 1> jj
      = Array<double, -1, 1>::LinSpaced(1.0, 0, static_cast<double>(M));
  double xstar = x[std::floor(N / 4 + 0.5)];  // first quartile of sample
  auto theta = 1.0 / x[N] + (1.0 - (M / (jj - 0.5)).sqrt()) / prior / xstar;
  auto l_theta = static_cast<double>(N) * lx(theta, x);  // profile log-lik
  auto w_theta
      = (l_theta - stan::math::log_sum_exp(l_theta)).exp();  // normalize
  auto theta_hat = (theta * w_theta).sum();
  auto k = (-theta_hat * x).log1p().mean();
  auto sigma = -k / theta_hat;
  // auto k = adjust_k_wip(k, n = N);
  constexpr double a = 10;
  const double n_plus_a = N + a;
  k = k * N / n_plus_a + a * 0.5 / n_plus_a;
  return std::make_tuple(k, sigma)
}

inline auto qgpd(const Eigen::Array<double, -1, 1>& p, double k, double sigma) {
  /*
    if (is.nan(sigma) || sigma <= 0) {
      return(rep(NaN, length(p)))
    }
  */
  return (sigma * (-k * log1p(-p)).expm1() / k).eval();
}

inline auto psis_smooth_tail(const Eigen::Array<double, -1, 1>& x,
                             double cutoff) {
  const auto x_size = x.size();
  const auto exp_cutoff = std::exp(cutoff);

  // save time not sorting since x already sorted
  auto fit = gpdfit(x.array().exp() - exp_cutoff);
  double k = std::get<0>(fit);
  double sigma = std::get<1>(fit);
  if (is_finite(k)) {
    auto p = (seq_len(len) - 0.5) / x_size;
    auto qq = qgpd(p, k, sigma) + exp_cutoff;
    return std::make_tuple(qq.log().eval(), k);
  } else {
    return std::make_tuple(x, k);
  }
}

inline auto max_n_element(const Eigen::Array<double, -1, 1>& lw_i,
                          size_t tail_len_i) {
  Eigen::Array<double, -1, 1> top_n = lw_i.head(tail_len_i);
  Eigen::Array<Eigen::Index, -1, 1> top_n_idx(tail_len_idx);
  return std::make_tuple(std::move(top_n), std::move(top_n_idx));
}

inline auto get_psis_weights(const Eigen::VectorXd& log_ratios_i,
                             size_t tail_len_i) {
  const auto S = log_ratios_i.size();
  // shift log ratios for safer exponentation
  const double max_log_ratio = log_ratios_i.maxCoeff();
  Eigen::Array<double, -1, 1> lw_i = log_ratios_i - max_log_ratio;
  double khat = 0;

  if (tail_len_i >= 5) {
    /*
    ord = sort.int(lw_i, index.return = TRUE)
    tail_ids = seq(S - tail_len_i + 1, S)
    lw_tail = ord$x[tail_ids]
    */
    // Get back tail + smallest but not on tail in ascending order
    auto max_n = max_n_elements(lw_i, tail_len_i + 1);
    Eigen::Array<double, -1, 1> lw_tail = std::get<0>(max_n);
    if (abs(max(lw_tail) - min(lw_tail))
        <= std::numeric_limits<double>::min() * 10) {
      /*
      warning(
        "Can't fit generalized Pareto distribution ",
        "because all tail values are the same.",
        call. = FALSE
      )
      */
    } else {
      // cutoff = ord$x[min(tail_ids) - 1] // largest value smaller than tail
      // values
      auto smoothed = psis_smooth_tail(lw_tail, lw_tail.head(1));
      auto khat = std::get<1>(smoothed);
      //         lw_i[ord$ix[tail_ids]] = smoothed$tail
      insert_smooth_to_tail(lw_i, std::get<1>(max_n), std::get<0>(smoothed));
    }
  }

  // truncate at max of raw wts (i.e., 0 since max has been subtracted)
  for (Eigen::Index i = 0; i < lw_i.size(); ++i) {
    if (lw_i(i) < 0) {
      lw_i(i) = 0.0;
    }
    // shift log weights back so that the smallest log weights remain unchanged
    // lw_i = lw_i + max(log_ratios_i)
  }
  return (lw_i - math::log_sum_exp(lw_i + max_log_ratio)).exp().eval();
}
}  // namespace psis
}  // namespace services
}  // namespace stan

#endif
