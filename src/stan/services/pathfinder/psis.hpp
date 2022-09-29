#ifndef STAN_SERVICES_PSIS_HPP
#define STAN_SERVICES_PSIS_HPP

#include <stan/math.hpp>
#include <stan/callbacks/logger.hpp>
#include <stan/services/error_codes.hpp>
#include <tbb/parallel_invoke.h>

namespace stan {
namespace services {
namespace psis {
namespace internal {

/**
 * Compute log joint likelihood parameter estimates from generalized pareto
 * distribution and the samples the parameters were estimated from.
 * @tparam EigArray1 An Eigen type inheriting from `ArrayBase` with dynamic
 * compile time rows and 1 compile time column.
 * @tparam EigArray2 An Eigen type inheriting from `ArrayBase` with dynamic
 * compile time rows and 1 compile time column.
 * @param[in] theta Estimates from generalized pareto distribution estimation
 * @param[in] x The sample that the parameters were estimated from.
 * @return Array of the joint log likelihood of parameter estimates from
 * generalized pareto distribution and the samples the parameters were estimated
 * from.
 */
template <typename EigArray1, typename EigArray2>
inline Eigen::Array<double, -1, 1> profile_loglikelihood(const EigArray1& theta,
                                                         const EigArray2& x) {
  Eigen::Array<double, -1, 1> k = ((-theta).matrix() * x.matrix().transpose())
                                      .array()
                                      .log1p()
                                      .matrix()
                                      .rowwise()
                                      .mean()
                                      .array();
  return (-theta / k).log() - k - 1;
}

/**
 * Estimate parameters of the Generalized Pareto distribution
 *
 * Given a sample `x`, Estimate the parameters `k` and $\sigma$ of
 * the Generalized Pareto Distribution (GPD), assuming the location parameter is
 * 0. By default the fit uses a prior for `k`, which will stabilize
 * estimates for very small sample sizes (and low effective sample sizes in the
 * case of MCMC samples). The weakly informative prior is a Gaussian prior
 * centered at 0.5.
 *
 * @tparam EigArray An Eigen type inheriting from `ArrayBase` with dynamic
 * compile time rows and 1 compile time column.
 * @param[in] x A numeric vector. The sample from which to estimate the
 * parameters.
 * @param[in] min_grid_pts The minimum number of grid points used in the fitting
 *   algorithm.
 * @return A pair of doubles with the first element `sigma` and the second
 * element `k`.
 *
 * @details Here the parameter `k is the negative of `k` in Zhang & Stephens
 * (2009).
 *
 * @references
 * Zhang, J., and Stephens, M. A. (2009). A new and efficient estimation method
 * for the generalized Pareto distribution. *Technometrics* **51**, 316-325.
 */
template <typename EigArray>
inline auto gpdfit(const EigArray& x, const Eigen::Index min_grid_pts = 30) {
  using array_vec_t = Eigen::Array<double, -1, 1>;
  constexpr auto prior = 3.0;
  const auto& x_ref = stan::math::to_ref(x);
  const Eigen::Index N = x_ref.size();
  // See section 4 of Zhang and Stephens (2009)
  const Eigen::Index M = min_grid_pts + std::floor(std::sqrt(N));
  auto linspaced_arr = array_vec_t::LinSpaced(M, 1, static_cast<double>(M));
  // first quartile of sample
  const double x_1st_qt = x_ref.coeff(
      static_cast<Eigen::Index>(std::floor(static_cast<double>(N) / 4.0 + 0.5))
      - 1l);
  array_vec_t theta
      = (1.0 / x_ref.coeff(N - 1)
         + (1.0 - (M / (linspaced_arr - 0.5)).sqrt()) / (prior * x_1st_qt));
  // profile log-lik
  array_vec_t l_theta
      = (static_cast<double>(N) * profile_loglikelihood(theta, x_ref));
  auto normalized_theta = (l_theta - stan::math::log_sum_exp(l_theta)).exp();
  const double theta_hat = (theta * normalized_theta).sum();
  double k = (-theta_hat * x_ref).log1p().mean();
  const double sigma = -k / theta_hat;
  constexpr double a = 10;
  const double n_plus_a = N + a;
  k = k * N / n_plus_a + a * 0.5 / n_plus_a;
  return std::make_pair(sigma, k);
}

/**
 * Inverse CDF of generalized pareto distribution
 * (assuming location parameter is 0)
 *
 * @tparam EigArray An Eigen type inheriting from `ArrayBase` with dynamic
 * compile time rows and 1 compile time column.
 * @param[in] p Vector of probabilities.
 * @param[in] k Scalar shape parameter.
 * @param[in] sigma Scalar scale parameter.
 * @return Vector of quantiles.
 */
template <typename EigArray>
inline auto qgpd(const EigArray& p, const double k, const double sigma) {
  return (sigma * stan::math::expm1(-k * (-p).log1p()) / k);
}

/**
 * PSIS tail smoothing for a single vector
 *
 * @tparam EigArray An Eigen type inheriting from `ArrayBase` with dynamic
 * compile time rows and 1 compile time column.
 * @param[in] x Array of tail elements already sorted in ascending order.
 * @param[in] cutoff
 * @return A pair containing:
 * `first`: Eigen Array same size as `x` containing the logs of the
 *   order statistics of the generalized pareto distribution.
 * `second`: scalar shape parameter estimate.
 */
template <typename EigArray>
inline auto psis_smooth_tail(const EigArray& x, const double cutoff) {
  const double exp_cutoff = std::exp(cutoff);
  const auto fit = gpdfit(x.array().exp() - exp_cutoff);
  const double k = fit.second;
  if (!std::isinf(k)) {
    const Eigen::Index x_size = x.size();
    const double sigma = fit.first;
    auto p = (Eigen::Array<double, -1, 1>::LinSpaced(x_size, 1, x_size) - 0.5)
             / x_size;
    return std::make_pair((qgpd(p, k, sigma) + exp_cutoff).log().eval(), k);
  } else {
    return std::make_pair(x.eval(), k);
  }
}

/**
 * This function takes the last element as pivot, places the pivot element at
 * its correct position in the sorted array, and places all values smaller than
 * the pivot to the left of the pivot and all greater elements to the right of
 * the pivot
 * @param[in, out] arr The Array of doubles to be sorted
 * @param[in, out] idx The index of the original positions of the elements of
 * `arr`. This is also sorted to keep track of the original positions of the
 * elements in `arr`.
 * @param[in] low Starting index of the sort
 * @param[in] high Ending index of the sort
 */
inline Eigen::Index quick_sort_partition(Eigen::Array<double, -1, 1>& arr,
                                         Eigen::Array<Eigen::Index, -1, 1>& idx,
                                         const Eigen::Index low,
                                         const Eigen::Index high) {
  const double pivot = arr.coeff(high);  // pivot
  Eigen::Index i = (low - 1l);           // Index of smaller element
  for (Eigen::Index j = low; j <= high - 1; ++j) {
    // If current element is smaller than or
    // equal to pivot
    if (arr.coeff(j) <= pivot) {
      ++i;  // increment index of smaller element
      std::swap(arr.coeffRef(i), arr.coeffRef(j));
      std::swap(idx.coeffRef(i), idx.coeffRef(j));
    }
  }
  std::swap(arr.coeffRef(i + 1l), arr.coeffRef(high));
  std::swap(idx.coeffRef(i + 1l), idx.coeffRef(high));
  return (i + 1l);
}

/**
 * Runs quick_sort optionally in parallel
 * @tparam Parallel Whether to allow the algorithm to attempt to run in
 * parallel.
 * @param[in, out] arr The Array of doubles to be sorted
 * @param[in, out] idx The index of the original positions of the elements of
 * `arr`. This is also sorted to keep track of the original positions of the
 * elements in `arr`.
 * @param[in] low Starting index of the sort
 * @param[in] high Ending index of the sort
 */
template <bool DoParallel = true>
inline void quick_sort(Eigen::Array<double, -1, 1>& arr,
                       Eigen::Array<Eigen::Index, -1, 1>& idx,
                       const Eigen::Index low, const Eigen::Index high) {
  if (low < high) {
    // pi is partitioning index, arr[p] is now at right place
    const Eigen::Index partition_idx
        = quick_sort_partition(arr, idx, low, high);
    // Separately sort elements before partition and after partition
    if (DoParallel && (high - low >= 400l)) {
      tbb::parallel_invoke(
          [&arr, &idx, low, partition_idx]() {
            quick_sort(arr, idx, low, partition_idx - 1);
          },
          [&arr, &idx, high, partition_idx]() {
            quick_sort(arr, idx, partition_idx + 1, high);
          });
    } else {
      quick_sort<false>(arr, idx, low, partition_idx - 1);
      quick_sort<false>(arr, idx, partition_idx + 1, high);
    }
  }
}

/**
 * Runs quick_sort optionally in parallel
 * @param[in, out] arr The Array of doubles to be sorted
 * @param[in, out] idx The index of the original positions of the elements of
 * `arr`. This is also sorted to keep track of the original positions of the
 * elements in `arr`.
 */
inline void quick_sort(Eigen::Array<double, -1, 1>& arr,
                       Eigen::Array<Eigen::Index, -1, 1>& idx) {
  if (arr.size() >= 400l) {
    quick_sort(arr, idx, 0l, arr.size() - 1l);
  } else {
    quick_sort<false>(arr, idx, 0l, arr.size() - 1l);
  }
}

inline auto largest_insertion(const Eigen::Array<double, -1, 1>& top_n,
                              const double value) {
  const Eigen::Index top_size = top_n.size();
  Eigen::Index low_idx = -1l;
  Eigen::Index high_idx = top_size;
  for (Eigen::Index low_idx = -1, probe_idx = (-1l + top_size) / 2l;
       high_idx - low_idx > 1l; probe_idx = (low_idx + high_idx) / 2l) {
    if (top_n.coeff(probe_idx) > value) {
      high_idx = probe_idx;
    } else {
      low_idx = probe_idx;
    }
  }
  return high_idx - 1l;
}

/**
 * Get the largest N elements of an array.
 * @param arr The normalized log ratios to sort
 * @param top_size The length of the tail that is needs to be sorted.
 * @return A pair with the largest N elements in `first` and the original index
 * of the largest N elements in `second`
 */
inline std::pair<Eigen::Array<double, -1, 1>, Eigen::Array<Eigen::Index, -1, 1>>
largest_n_elements(const Eigen::Array<double, -1, 1>& arr,
                   const Eigen::Index top_size) {
  Eigen::Array<double, -1, 1> top_n = arr.head(top_size);
  Eigen::Array<Eigen::Index, -1, 1> top_n_idx
      = Eigen::Array<Eigen::Index, -1, 1>::LinSpaced(top_size, 0, top_size);
  quick_sort(top_n, top_n_idx);
  for (Eigen::Index i = top_size; i < arr.size(); ++i) {
    if (arr.coeff(i) >= top_n.coeff(0)) {
      const Eigen::Index starting_pos = largest_insertion(top_n, arr.coeff(i));
      for (Eigen::Index k = 1; k <= starting_pos; ++k) {
        top_n.coeffRef(k - 1) = top_n.coeff(k);
      }
      top_n.coeffRef(starting_pos) = arr.coeff(i);
      for (Eigen::Index k = 1; k <= starting_pos; ++k) {
        top_n_idx.coeffRef(k - 1) = top_n_idx.coeff(k);
      }
      top_n_idx.coeffRef(starting_pos) = i;
    }
  }
  return std::make_pair(std::move(top_n), std::move(top_n_idx));
}
}  // namespace internal

/*
 * Compute Pareto smoothed importance sampling (PSIS) log weights.
 *
 * @tparam EigArray An Eigen type inheriting from `ArrayBase` with dynamic
 * compile time rows and 1 compile time column.
 * @param[in] log_ratios Array of logarithms of importance ratios
 * @param[in] tail_len Size of the tail
 */
template <typename EigArray, typename Logger>
inline Eigen::Array<double, -1, 1> psis_weights(const EigArray& log_ratios,
                                                Eigen::Index tail_len,
                                                Logger& logger) {
  const auto S = log_ratios.size();
  // shift log ratios for safer exponentation
  const double max_log_ratio = log_ratios.maxCoeff();
  Eigen::Array<double, -1, 1> llr_weights = log_ratios.array() - max_log_ratio;
  if (tail_len >= 5) {
    // Get back tail + smallest but not on tail in ascending order
    std::pair<Eigen::Array<double, -1, 1>, Eigen::Array<Eigen::Index, -1, 1>>
        max_n = internal::largest_n_elements(llr_weights, tail_len + 1);
    auto lw_tail = max_n.first.tail(tail_len);
    double cutoff = max_n.first(0);
    if (unlikely(lw_tail.maxCoeff() - lw_tail.minCoeff()
                 <= std::numeric_limits<double>::min() * 10)) {
      double eps_diff = lw_tail.maxCoeff() - lw_tail.minCoeff();
      logger.warn(
       std::string("In PSIS Weight Calculation: Difference "
       "between the tails is ") +
        std::to_string(eps_diff) +
        "which is too small for estimating the generalized pareto values."
        " Returning non-pareto smoothed weights.");
    } else {
      auto smoothed = internal::psis_smooth_tail(lw_tail, cutoff);
      auto idx = max_n.second.tail(tail_len);
      const Eigen::Index idx_size = idx.size();
      for (Eigen::Index i = 0; i < idx_size; ++i) {
        llr_weights.coeffRef(idx.coeff(i)) = smoothed.first.coeff(i);
      }
      if (smoothed.second > 0.7) {
        logger.warn(std::string("Pareto k value (") +
         std::to_string(smoothed.second) +
         ") is greater than 0.7 which often indicates model"
         " misspecification.");
      }
    }
  }

  // truncate at max of raw wts (i.e., 0 since max has been subtracted)
  for (Eigen::Index i = 0; i < llr_weights.size(); ++i) {
    if (llr_weights.coeff(i) > 0) {
      llr_weights.coeffRef(i) = 0.0;
    }
  }
  auto max_adj = (llr_weights + max_log_ratio).eval();
  return (max_adj - stan::math::log_sum_exp(max_adj)).exp();
}

}  // namespace psis
}  // namespace services
}  // namespace stan

#endif
