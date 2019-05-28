#ifndef STAN_ANALYZE_MCMC_COMPUTE_EFFECTIVE_SAMPLE_SIZE_HPP
#define STAN_ANALYZE_MCMC_COMPUTE_EFFECTIVE_SAMPLE_SIZE_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/analyze/mcmc/autocovariance.hpp>
#include <stan/analyze/mcmc/split_chains.hpp>
#include <algorithm>
#include <cmath>
#include <vector>

namespace stan {
namespace analyze {

  /**
   * Returns the effective sample size for the specified parameter
   * across all kept samples.
   *
   * See more details in Stan reference manual section "Effective
   * Sample Size". http://mc-stan.org/users/documentation
   *
   * Current implementation assumes chains are all of equal size and
   * draws are stored in contiguous blocks of memory.
   *
   * @param Eigen::VectorXd stores autocovariance of each chain
   * @param Eigen::VectorXd stores means of each chain
   * @return effective sample size for the specified parameter
   */
  template <typename DerivedA, typename DerivedB>
  inline
  double effective_sample_size_impl(const Eigen::MatrixBase<DerivedA>& acov,
                                    const Eigen::MatrixBase<DerivedB>& chain_mean) {

    int num_chains = chain_mean.size();
    Eigen::VectorXd chain_var(num_chains);
    for (int chain = 0; chain < num_chains; ++chain) {
      chain_var(chain) = acov(chain)(0);
    }

    // need to generalize to each jagged draws per chain
    int num_draws = acov(0).size();
    double mean_var = chain_var.mean() * num_draws / (num_draws - 1);
    double var_plus = mean_var * (num_draws - 1) / num_draws;
    if (num_chains > 1)
      var_plus += (chain_mean.array() - chain_mean.mean()).square().sum() / (num_chains - 1);

    Eigen::VectorXd rho_hat_s(num_draws);
    rho_hat_s.setZero();
    Eigen::VectorXd acov_s(num_chains);
    for (int chain = 0; chain < num_chains; ++chain)
      acov_s(chain) = acov(chain)(1);
    double rho_hat_even = 1.0;
    rho_hat_s(0) = rho_hat_even;
    double rho_hat_odd = 1 - (mean_var - acov_s.mean()) / var_plus;
    rho_hat_s(1) = rho_hat_odd;
    // Geyer's initial positive sequence
    size_t s = 1;
    for (; s < (num_draws - 4) && (rho_hat_even + rho_hat_odd) > 0;
         s += 2) {
      for (int chain = 0; chain < num_chains; ++chain)
        acov_s(chain) = acov(chain)(s + 1);
      rho_hat_even = 1 - (mean_var - acov_s.mean()) / var_plus;
      for (int chain = 0; chain < num_chains; ++chain)
        acov_s(chain) = acov(chain)(s + 2);
      rho_hat_odd = 1 - (mean_var - acov_s.mean()) / var_plus;
      if ((rho_hat_even + rho_hat_odd) >= 0) {
        rho_hat_s(s + 1) = rho_hat_even;
        rho_hat_s(s + 2) = rho_hat_odd;
      }
    }

    int max_s = s;
    // this is used in the improved estimate
    if (rho_hat_even > 0)
      rho_hat_s(max_s + 1) = rho_hat_even;

    // Geyer's initial monotone sequence
    for (int s = 1; s <= max_s - 3; s += 2) {
      if (rho_hat_s(s + 1) + rho_hat_s(s + 2) >
          rho_hat_s(s - 1) + rho_hat_s(s)) {
        rho_hat_s(s + 1) = (rho_hat_s(s - 1) + rho_hat_s(s)) / 2;
        rho_hat_s(s + 2) = rho_hat_s(s + 1);
      }
    }

    double ess = num_chains * num_draws;
    // Geyer's truncated estimate
    // Improved estimate reduces variance in antithetic case
    double tau_hat = -1 + 2 * rho_hat_s.head(max_s).sum() + rho_hat_s(max_s + 1);
    // Safety check for negative values and with max ess equal to ess*log10(ess)
    return ess / std::max(tau_hat, 1 / std::log10(ess));
  }

  /**
   * Returns the effective sample size for the specified parameter
   * across all kept samples.
   *
   * See more details in Stan reference manual section "Effective
   * Sample Size". http://mc-stan.org/users/documentation
   *
   * Current implementation assumes chains are all of equal size and
   * draws are stored in contiguous blocks of memory.
   *
   * @param Eigen::MatrixBase stores arrays of chains
   * @return effective sample size for the specified parameter
   */
  template <typename Derived>
  inline
  double compute_effective_sample_size(const Eigen::MatrixBase<Derived>& draws) {

    int num_chains = draws.cols();
    Eigen::Matrix<Eigen::VectorXd, Eigen::Dynamic, 1> acov(num_chains);
    Eigen::VectorXd chain_mean(num_chains);
    for (int chain = 0; chain < num_chains; ++chain) {
      autocovariance<double>(draws.col(chain), acov(chain));
      chain_mean(chain) = draws.col(chain).mean();
    }

    return effective_sample_size_impl(acov, chain_mean);
  }

    /**
     * Returns the effective sample size for the specified parameter
     * across all kept samples.
     *
     * See more details in Stan reference manual section "Effective
     * Sample Size". http://mc-stan.org/users/documentation
     *
     * Current implementation assumes chains are all of equal size and
     * draws are stored in contiguous blocks of memory.
     *
     * @param std::vector stores pointers to arrays of chains
     * @param std::vector stores sizes of chains
     * @return effective sample size for the specified parameter
     */
    inline
    double compute_effective_sample_size(std::vector<const double*> draws,
                                         std::vector<size_t> sizes) {

      // std::cout << compute_effective_sample_size(split_chains(draws, sizes))
      //           << std::endl;

      int num_chains = sizes.size();
      Eigen::Matrix<Eigen::VectorXd, Eigen::Dynamic, 1> acov(num_chains);
      Eigen::VectorXd chain_mean(num_chains);
      for (int chain = 0; chain < num_chains; ++chain) {
        Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 1>>
          draw(draws[chain], sizes[chain]);
        autocovariance<double>(draw, acov(chain));
        chain_mean(chain) = draw.mean();
      }

      return effective_sample_size_impl(acov, chain_mean);
    }

    inline
    double compute_effective_sample_size(std::vector<const double*> draws,
                                         size_t size) {
      int num_chains = draws.size();
      std::vector<size_t> sizes(num_chains, size);
      return compute_effective_sample_size(draws, sizes);
    }
} // namespace analyze
} // namespace stan

#endif
