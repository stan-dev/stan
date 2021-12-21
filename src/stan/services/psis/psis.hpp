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
#include <tbb/parallel_invoke.h>
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

template <typename T>
inline bool is_finite(T x) noexcept {
    return x != std::numeric_limits<T>::infinity();
}

template <typename T>
inline auto log_sum_exp(const T& v) {
    if (unlikely(v.size() == 0)) {
        return -std::numeric_limits<double>::infinity();
    } else {
      const auto& v_ref = stan::math::to_ref(v);
      const double max = v_ref.maxCoeff();
      if (!std::isfinite(max)) {
          return max;
      }
      return max + std::log((v_ref.array() - max).exp().sum());
    }
}

template <typename EigArray1, typename EigArray2>
inline Eigen::Array<double, -1, 1> lx(const EigArray1& a, const EigArray2& x) {
    Eigen::Array<double, -1, 1> k = ((-a).matrix() * x.matrix().transpose()).array().log1p().matrix().rowwise().mean().array();
    return (-a / k).log() - k - 1;
}


template <typename EigArray>
inline auto gpdfit(const EigArray& x, bool wip = true, size_t min_grid_pts = 30) {
    // See section 4 of Zhang and Stephens (2009)
    const auto N = x.size();
    constexpr auto prior = 3.0;
    const auto M = min_grid_pts + std::floor(std::sqrt(N));
    Eigen::Array<double, -1, 1> jj =
        Eigen::Array<double, -1, 1>::LinSpaced(M, 1, static_cast<double>(M));
    double xstar = x[std::floor(N / 4 + 0.5) - 1];  // first quartile of sample
    auto theta = (1.0 / x[N - 1] + (1.0 - (M / (jj - 0.5)).sqrt()) / prior / xstar).eval();
    auto l_theta = (static_cast<double>(N) * lx(theta, x)).eval();   // profile log-lik
    auto w_theta = (l_theta - log_sum_exp(l_theta)).exp();  // normalize
    auto theta_hat = (theta * w_theta).sum();
    double k = (-theta_hat * x).log1p().mean();
    auto sigma = -k / theta_hat;
    constexpr double a = 10;
    const double n_plus_a = N + a;
    k = k * N / n_plus_a + a * 0.5 / n_plus_a;
    return std::make_tuple(k, sigma);
}

template <typename EigArray>
inline auto qgpd(const EigArray& p, double k, double sigma) {
    return (sigma * stan::math::expm1(-k * (-p).log1p()) / k).eval();
}


template <typename EigArray>
inline auto psis_smooth_tail(const EigArray& x, double cutoff) {
    const auto x_size = x.size();
    const auto exp_cutoff = std::exp(cutoff);

    // save time not sorting since x already sorted
    auto fit = gpdfit(x.array().exp() - exp_cutoff);
    double k = std::get<0>(fit);
    double sigma = std::get<1>(fit);
    if (is_finite(k)) {
        auto p =
            (Eigen::Array<double, -1, 1>::LinSpaced(x_size, 1, x_size) - 0.5) /
            x_size;
        Eigen::Array<double, -1, 1> blah = qgpd(p, k, sigma);
        return std::make_tuple((blah + exp_cutoff).log().eval(), k);
    } else {
        return std::make_tuple(x, k);
    }
}

/* This function takes last element as pivot, places
   the pivot element at its correct position in sorted
    array, and places all smaller (smaller than pivot)
   to left of pivot and all greater elements to right
   of pivot */
template <typename EigDblArr, typename EigIdxArr>
inline Eigen::Index quick_sort_partition(EigDblArr&& arr, EigIdxArr&& idx, const Eigen::Index low, const Eigen::Index high) {
    const auto pivot = arr[high];      // pivot
    Eigen::Index i = (low - 1);  // Index of smaller element

    for (Eigen::Index j = low; j <= high - 1; ++j) {
        // If current element is smaller than or
        // equal to pivot
        if (arr[j] <= pivot) {
            ++i;  // increment index of smaller element
            std::swap(arr.coeffRef(i), arr.coeffRef(j));
            std::swap(idx.coeffRef(i), idx.coeffRef(j));
        }
    }
    std::swap(arr.coeffRef(i + 1), arr.coeffRef(high));
    std::swap(idx.coeffRef(i + 1), idx.coeffRef(high));
    return (i + 1);
}

/* The main function that implements quick_sort
 arr[] --> Array to be sorted,
  low  --> Starting index,
  high  --> Ending index */
template <typename EigDblArr, typename EigIdxArr>
inline void quick_sort(EigDblArr&& arr, EigIdxArr&& idx, const Eigen::Index low, const Eigen::Index high) {
    if (low < high) {
        /* pi is partitioning index, arr[p] is now
           at right place */
        const Eigen::Index pi = quick_sort_partition(arr, idx, low, high);

        // Separately sort elements before
        // partition and after partition
        if (high - low >= 400) {
          tbb::parallel_invoke( [&]{quick_sort(arr, idx, low, pi - 1);},
                               [&]{quick_sort(arr, idx, pi + 1, high);} );
        } else {
          quick_sort(arr, idx, low, pi - 1);
          quick_sort(arr, idx, pi + 1, high);
        }
    }
}

template <typename EigDblArr, typename EigIdxArr>
inline void quick_sort(EigDblArr&& arr, EigIdxArr&& idx) {
    quick_sort(arr, idx, 0, arr.size() - 1);
}

template <typename T>
inline auto max_n_insertion_start(T&& top_n, const double value) {
  const Eigen::Index top_size = top_n.size();
  Eigen::Index low_idx = -1l;
  Eigen::Index high_idx = top_size;
  while (high_idx - low_idx > 1l) {
      const Eigen::Index probe_idx = (low_idx + high_idx) / 2l;
      const double curr_val = top_n.coeff(probe_idx);
      if (curr_val > value) {
        high_idx = probe_idx;
      } else {
        low_idx = probe_idx;
      }
  }
  if (high_idx == top_size) {
    return top_size - 1;
  } else {
    return high_idx - 1;
  }
}

template <typename EigArray>
inline auto max_n_elements(const EigArray& lw_i, const size_t tail_len_i) {
    Eigen::Array<double, -1, 1> top_n = lw_i.head(tail_len_i);
    Eigen::Array<Eigen::Index, -1, 1> top_n_idx =
        Eigen::Array<Eigen::Index, -1, 1>::LinSpaced(tail_len_i, 0, tail_len_i);
    quick_sort(top_n, top_n_idx);
    for (Eigen::Index i = tail_len_i; i < tail_len_i; ++i) {
      if (lw_i.coeff(i) >= top_n.coeffRef(0)) {
        const Eigen::Index starting_pos = max_n_insertion_start(top_n, lw_i.coeff(i));
        for (Eigen::Index k = 1; k <= starting_pos; ++k) {
            top_n.coeffRef(k - 1) = top_n.coeff(k);
        }
        top_n.coeffRef(starting_pos) = lw_i.coeff(i);
        for (Eigen::Index k = 1; k <= starting_pos; ++k) {
            top_n_idx.coeffRef(k - 1) = top_n_idx.coeff(k);
        }
        top_n_idx.coeffRef(starting_pos) = i;
      }
    }
    return std::make_tuple(std::move(top_n), std::move(top_n_idx));
}

template <typename EigDblArray1, typename EigIndexArray, typename EigDblArray2>
inline void insert_smooth_to_tail(EigDblArray1&& lw_i,
                                  EigIndexArray&& idx,
                                  EigDblArray2&& smoothed) {
    const Eigen::Index idx_size = idx.size();
    for (Eigen::Index i = 0; i < idx_size; ++i) {
        lw_i.coeffRef(idx.coeff(i)) = smoothed.coeff(i);
    }
}

template <typename EigArray>
inline auto get_psis_weights(const EigArray& log_ratios_i,
                             size_t tail_len_i) {
    const auto S = log_ratios_i.size();
    // shift log ratios for safer exponentation
    const double max_log_ratio = log_ratios_i.maxCoeff();
    Eigen::Array<double, -1, 1> lw_i = log_ratios_i - max_log_ratio;
    if (tail_len_i >= 5) {
        // Get back tail + smallest but not on tail in ascending order
        auto max_n = max_n_elements(lw_i, tail_len_i + 1);
        Eigen::Array<double, -1, 1> lw_tail = std::get<0>(max_n).tail(tail_len_i);
        double cutoff = std::get<0>(max_n)(0);
        if (lw_tail.maxCoeff() - lw_tail.minCoeff() <=
            std::numeric_limits<double>::min() * 10) {
             // I need to throw a warning here
        } else {
            auto smoothed = psis_smooth_tail(lw_tail, cutoff);
            auto khat = std::get<1>(smoothed);
            insert_smooth_to_tail(lw_i, std::get<1>(max_n).tail(tail_len_i),
                                  std::get<0>(smoothed));
        }
    }

    // truncate at max of raw wts (i.e., 0 since max has been subtracted)
    for (Eigen::Index i = 0; i < lw_i.size(); ++i) {
        if (lw_i.coeff(i) > 0) {
            lw_i.coeffRef(i) = 0.0;
        }
    }
    auto max_adj = (lw_i + max_log_ratio).eval();
    return (max_adj - log_sum_exp(max_adj)).exp().eval();
}

}  // namespace psis
}  // namespace services
}  // namespace stan

#endif
