#ifndef STAN_MCMC_HMC_HAMILTONIANS_DIAG_E_POINT_HPP
#define STAN_MCMC_HMC_HAMILTONIANS_DIAG_E_POINT_HPP

#include <stan/callbacks/writer.hpp>
#include <stan/math/prim/err.hpp>
#include <stan/mcmc/hmc/hamiltonians/ps_point.hpp>

namespace stan {
namespace mcmc {
/**
 * Point in a phase space with a base
 * Euclidean manifold with diagonal metric
 */
class diag_e_point : public ps_point {
 private:
  /**
   * Vector of diagonal elements of inverse mass matrix.
   */
  Eigen::VectorXd inv_e_metric_;

 public:
  /**
   * Construct a diag point in n-dimensional phase space
   * with vector of ones for diagonal elements of inverse mass matrix.
   *
   * @param n number of dimensions
   */
  explicit diag_e_point(int n) : ps_point(n), inv_e_metric_(n) {
    inv_e_metric_.setOnes();
  }

  /**
   * Set elements of mass matrix
   *
   * @param inv_e_metric initial mass matrix
   */
  template <typename EigVec, require_eigen_vector_t<EigVec>* = nullptr>
  void set_inv_metric(EigVec&& inv_e_metric) {
    inv_e_metric_ = std::forward<EigVec>(inv_e_metric);
  }

  /**
   * Get inverse metric
   *
   * @return reference to the inverse metric
   */
  const Eigen::VectorXd& get_inv_metric() const { return inv_e_metric_; }

  /**
   * Write elements of mass matrix to string and handoff to writer.
   *
   * @param writer Stan writer callback
   */
  inline void write_metric(stan::callbacks::writer& writer) {
    writer("Diagonal elements of inverse mass matrix:");
    std::stringstream inv_e_metric_ss;
    inv_e_metric_ss << inv_e_metric_(0);
    for (int i = 1; i < inv_e_metric_.size(); ++i)
      inv_e_metric_ss << ", " << inv_e_metric_(i);
    writer(inv_e_metric_ss.str());
  }
};

}  // namespace mcmc
}  // namespace stan

#endif
