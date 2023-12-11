#ifndef STAN_MCMC_HMC_HAMILTONIANS_UNIT_E_POINT_HPP
#define STAN_MCMC_HMC_HAMILTONIANS_UNIT_E_POINT_HPP

#include <stan/callbacks/writer.hpp>
#include <stan/mcmc/hmc/hamiltonians/ps_point.hpp>

namespace stan {
namespace mcmc {
/**
 * Point in a phase space with a base
 * Euclidean manifold with unit metric
 */
class unit_e_point : public ps_point {
 public:
  /**
   * Vector of diagonal elements of inverse mass matrix.
   */
  Eigen::VectorXd inv_e_metric_;

  /**
   * Construct a diag point in n-dimensional phase space
   * with vector of ones for diagonal elements of inverse mass matrix.
   *
   * @param n number of dimensions
   */
  explicit unit_e_point(int n) : ps_point(n), inv_e_metric_(n) {
    inv_e_metric_.setOnes();
  }

  inline void write_metric(stan::callbacks::writer& writer) {
    writer("No free parameters for unit metric");
  }

  inline std::string metric_type() { return "unit_e"; }
};

}  // namespace mcmc
}  // namespace stan

#endif
