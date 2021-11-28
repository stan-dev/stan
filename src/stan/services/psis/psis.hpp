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

template <typename T>
inline bool is_finite(T x) {
    return x != std::numeric_limits<T>::infinity();
}

template <typename T>
inline auto log_sum_exp(const T& v) {
    if (v.size() == 0) {
        return -std::numeric_limits<double>::infinity();
    }
    const auto& v_ref = v;
    const double max = v_ref.maxCoeff();
    if (!std::isfinite(max)) {
        return max;
    }
    return max + std::log((v_ref.array() - max).exp().sum());
}

inline Eigen::Array<double, -1, 1> lx(const Eigen::Array<double, -1, 1>& a,
                                      const Eigen::Array<double, -1, 1>& x) {
    Eigen::Array<double, -1, 1> k = ((-a).matrix() * x.matrix().transpose()).array().log1p().matrix().rowwise().mean().array();
    return (-a / k).log() - k - 1;
}

inline auto gpdfit(const Eigen::Array<double, -1, 1>& x, bool wip = true,
            size_t min_grid_pts = 30) {
    // See section 4 of Zhang and Stephens (2009)
    const auto N = x.size();
    constexpr auto prior = 3.0;
    const auto M = min_grid_pts + std::floor(std::sqrt(N));
    Eigen::Array<double, -1, 1> jj =
        Eigen::Array<double, -1, 1>::LinSpaced(M, 1, static_cast<double>(M));
    double xstar = x[std::floor(N / 4 + 0.5) - 1];  // first quartile of sample
//    std::cout << "\nx: \n" << x;
//    std::cout << "\njj: \n" << jj;

    auto theta = (1.0 / x[N - 1] + (1.0 - (M / (jj - 0.5)).sqrt()) / prior / xstar).eval();
    auto l_theta = (static_cast<double>(N) * lx(theta, x)).eval();   // profile log-lik
//    std::cout << "\nx[N - 1]: " << x[N - 1] << "\n";
//    std::cout << "\nx_star: " << xstar << "\n";
//    std::cout << "\nl_theta: \n" << l_theta;
    auto w_theta = (l_theta - log_sum_exp(l_theta)).exp().eval();  // normalize
//    std::cout << "\ntheta: \n" << theta;
//    std::cout << "\nw_theta: \n" << w_theta << "\n";
    auto theta_hat = (theta * w_theta).sum();
    auto k = (-theta_hat * x).log1p().mean();
    auto sigma = -k / theta_hat;
    // auto k = adjust_k_wip(k, n = N);
    constexpr double a = 10;
    const double n_plus_a = N + a;
    k = k * N / n_plus_a + a * 0.5 / n_plus_a;
//    std::cout << "\nk: " << k;
//    std::cout << "\nsigma: " << sigma;
    return std::make_tuple(k, sigma);
}

inline auto qgpd(const Eigen::Array<double, -1, 1>& p, double k, double sigma) {
    /*
      if (is.nan(sigma) || sigma <= 0) {
        return(rep(NaN, length(p)))
      }
    */
    // TODO: Replace with expm1()
    return (sigma * stan::math::expm1(-k * (-p).log1p()) / k).eval();
}

inline auto psis_smooth_tail(const Eigen::Array<double, -1, 1>& x,
                             double cutoff) {
    const auto x_size = x.size();
    const auto exp_cutoff = std::exp(cutoff);

    // save time not sorting since x already sorted
    auto fit = gpdfit(x.array().exp() - exp_cutoff);
    double k = std::get<0>(fit);
    double sigma = std::get<1>(fit);
    std::cout << "\n k: " << k << "\n";
    std::cout << "\n sigma: " << sigma << "\n";
    if (is_finite(k)) {
        auto p =
            (Eigen::Array<double, -1, 1>::LinSpaced(x_size, 1, x_size) - 0.5) /
            x_size;
        Eigen::Array<double, -1, 1> blah = qgpd(p, k, sigma);
        std::cout << "\nblah: \n" << blah << "\n";
        auto qq = blah + exp_cutoff;
        std::cout << "\n p: \n" << p << "\n";
        std::cout << "\n qq: \n" << qq << "\n";
        return std::make_tuple(qq.log().eval(), k);
    } else {
        return std::make_tuple(x, k);
    }
}

void bubble_sort(Eigen::Array<double, -1, 1>& arr,
                 Eigen::Array<Eigen::Index, -1, 1>& arr_idx) {
    const auto n = arr.size();
    Eigen::Index i, j;
    for (i = 0; i < arr.size(); i++) {
        // Last i elements are already in place
        for (j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                std::swap(arr[j], arr[j + 1]);
                std::swap(arr_idx[j], arr_idx[j + 1]);
            }
        }
    }
}

inline auto max_n_elements(const Eigen::Array<double, -1, 1>& lw_i,
                           size_t tail_len_i) {
    Eigen::Array<double, -1, 1> top_n = lw_i.head(tail_len_i);
    Eigen::Array<Eigen::Index, -1, 1> top_n_idx =
        Eigen::Array<Eigen::Index, -1, 1>::LinSpaced(tail_len_i, 0, tail_len_i);
    bubble_sort(top_n, top_n_idx);
    for (Eigen::Index i = tail_len_i; i < lw_i.size(); ++i) {
        for (Eigen::Index j = top_n.size() - 1; j >= 0; --j) {
            if (top_n[j] <= lw_i[i]) {
                //              std::cout << "j: " << j;
                for (Eigen::Index k = 1; k <= j; ++k) {
                    //                  std::cout << " k: " << k;
                    top_n[k - 1] = top_n[k];
                    top_n_idx[k - 1] = top_n_idx[k];
                    //                std::cout << " top_n: " <<
                    //                top_n.transpose();
                }
                top_n[j] = lw_i[i];
                top_n_idx[j] = i;
                //              std::cout << "\n";
                break;
            }
        }
    }
    return std::make_tuple(std::move(top_n), std::move(top_n_idx));
}

inline void insert_smooth_to_tail(Eigen::Array<double, -1, 1>& lw_i,
                                  const Eigen::Array<Eigen::Index, -1, 1>& idx,
                                  const Eigen::Array<double, -1, 1>& smoothed) {
    for (Eigen::Index i = 0; i < idx.size(); ++i) {
        lw_i[idx[i]] = smoothed[i];
    }
}

inline auto get_psis_weights(const Eigen::Array<double, -1, 1>& log_ratios_i,
                             size_t tail_len_i) {
    const auto S = log_ratios_i.size();
    // shift log ratios for safer exponentation
    const double max_log_ratio = log_ratios_i.maxCoeff();
    Eigen::Array<double, -1, 1> lw_i = log_ratios_i - max_log_ratio;
    std::cout << "\nlw_i: \n" << lw_i << "\n";
    double khat = 0;

    if (tail_len_i >= 5) {
        /*
        ord = sort.int(lw_i, index.return = TRUE)
        tail_ids = seq(S - tail_len_i + 1, S)
        lw_tail = ord$x[tail_ids]
        */
        // Get back tail + smallest but not on tail in ascending order
        auto max_n = max_n_elements(lw_i, tail_len_i + 1);
        Eigen::Array<double, -1, 1> lw_tail = std::get<0>(max_n).tail(tail_len_i);
        double cutoff = std::get<0>(max_n)(0);
        std::cout << "\ncutoff: " << cutoff << "\n";
        std::cout << "\ntop_n: \n" << lw_tail << "\n";
        if (lw_tail.maxCoeff() - lw_tail.minCoeff() <=
            std::numeric_limits<double>::min() * 10) {
            /*
            warning(
              "Can't fit generalized Pareto distribution ",
              "because all tail values are the same.",
              call. = FALSE
            )
            */
        } else {
            // cutoff = ord$x[min(tail_ids) - 1] // largest value smaller than
            // tail values
            auto smoothed = psis_smooth_tail(lw_tail, cutoff);
            std::cout << "\n smoothed: \n" << std::get<0>(smoothed) << "\n";
            auto khat = std::get<1>(smoothed);
            std::cout << "\nkhat: " << khat << "\n";
            //         lw_i[ord$ix[tail_ids]] = smoothed$tail
            std::cout << "\nidx_n: \n" << std::get<1>(max_n).tail(tail_len_i) << "\n";
            insert_smooth_to_tail(lw_i, std::get<1>(max_n).tail(tail_len_i),
                                  std::get<0>(smoothed));
            std::cout << "\nnew lw_i: \n" << lw_i << "\n";
        }
    }

    // truncate at max of raw wts (i.e., 0 since max has been subtracted)
    for (Eigen::Index i = 0; i < lw_i.size(); ++i) {
        if (lw_i(i) > 0) {
            lw_i(i) = 0.0;
        }
        // shift log weights back so that the smallest log weights remain
        // unchanged lw_i = lw_i + max(log_ratios_i)
    }
    auto max_adj = (lw_i + max_log_ratio).eval();
    std::cout << "\nmax_adj: \n" << max_adj << "\n";
    return (max_adj - log_sum_exp(max_adj)).exp().eval();
}

}  // namespace psis
}  // namespace services
}  // namespace stan

#endif
