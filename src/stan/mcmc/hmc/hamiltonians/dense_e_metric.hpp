#ifndef STAN_MCMC_HMC_HAMILTONIANS_DENSE_E_METRIC_HPP
#define STAN_MCMC_HMC_HAMILTONIANS_DENSE_E_METRIC_HPP

#include <stan/callbacks/logger.hpp>
#include <stan/math/prim/mat.hpp>
#include <stan/mcmc/hmc/hamiltonians/base_hamiltonian.hpp>
#include <stan/mcmc/hmc/hamiltonians/dense_e_point.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/normal_distribution.hpp>
#include <Eigen/Cholesky>

namespace stan {
namespace mcmc {

// Euclidean manifold with dense metric
template <class Model, class BaseRNG>
class dense_e_metric : public base_hamiltonian<dense_e_metric<Model, BaseRNG>, Model, dense_e_point, BaseRNG> {
 public:
  explicit dense_e_metric(const Model& model)
      : base_hamiltonian<dense_e_metric<Model, BaseRNG>, Model, dense_e_point, BaseRNG>(model) {
        dtau_dq_ = Eigen::VectorXd::Zero(this->model_.num_params_r());
      }
  Eigen::VectorXd dtau_dq_;
  inline auto T(dense_e_point& z) {
    return 0.5 * z.p.transpose() * z.inv_e_metric_ * z.p;
  }

  inline auto tau(dense_e_point& z) { return T(z); }

  inline auto phi(dense_e_point& z) { return this->V(z); }

  inline auto dG_dt(dense_e_point& z, callbacks::logger& logger) {
    return 2 * T(z) - z.q.dot(z.g);
  }

  inline auto dtau_dq(dense_e_point& z, callbacks::logger& logger) {
    return dtau_dq_;
  }

  inline const auto dtau_dq(dense_e_point& z, callbacks::logger& logger) const {
    return dtau_dq_;
  }

  inline auto dtau_dp(dense_e_point& z) { return z.inv_e_metric_ * z.p; }

  inline auto& dphi_dq(dense_e_point& z, callbacks::logger& logger) {
    return z.g;
  }

  inline void sample_p(dense_e_point& z, BaseRNG& rng) {
    typedef typename stan::math::index_type<Eigen::VectorXd>::type idx_t;
    boost::variate_generator<BaseRNG&, boost::normal_distribution<> >
        rand_dense_gaus(rng, boost::normal_distribution<>());

    Eigen::VectorXd u(z.p.size());

    for (idx_t i = 0; i < u.size(); ++i)
      u(i) = rand_dense_gaus();

    z.p = z.inv_e_metric_.llt().matrixU().solve(u);
  }
};

}  // namespace mcmc
}  // namespace stan
#endif
