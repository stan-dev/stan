#ifndef STAN_MCMC_HMC_HAMILTONIANS_AUTO_E_POINT_HPP
#define STAN_MCMC_HMC_HAMILTONIANS_AUTO_E_POINT_HPP

#include <stan/callbacks/writer.hpp>
#include <stan/mcmc/hmc/hamiltonians/ps_point.hpp>

namespace stan {
namespace mcmc {
/**
 * Point in a phase space with a base
 * Euclidean manifold with auto metric
 */
class auto_e_point : public ps_point {
 public:
  /**
   * Inverse mass matrix.
   */
  Eigen::MatrixXd inv_e_metric_;

  /**
   * Is inv_e_metric_ diagonal or not
   */
  bool is_diagonal_;

  /**
   * Construct a auto point in n-dimensional phase space
   * with identity matrix as inverse mass matrix.
   *
   * @param n number of dimensions
   */
  explicit auto_e_point(int n)
      : ps_point(n), inv_e_metric_(n, n), is_diagonal_(true) {
    inv_e_metric_.setIdentity();
  }

  /**
   * Set elements of mass matrix
   *
   * @param inv_e_metric initial mass matrix
   */
  void set_metric(const Eigen::MatrixXd& inv_e_metric) {
    inv_e_metric_ = inv_e_metric;
    is_diagonal_ = false;
  }

  /**
   * Write elements of mass matrix to string and handoff to writer.
   *
   * @param writer Stan writer callback
   */
  inline void write_metric(stan::callbacks::writer& writer) {
    writer("Elements of inverse mass matrix:");
    for (int i = 0; i < inv_e_metric_.rows(); ++i) {
      std::stringstream inv_e_metric_ss;
      inv_e_metric_ss << inv_e_metric_(i, 0);
      for (int j = 1; j < inv_e_metric_.cols(); ++j)
        inv_e_metric_ss << ", " << inv_e_metric_(i, j);
      writer(inv_e_metric_ss.str());
    }
  }
};

}  // namespace mcmc
}  // namespace stan

#endif
