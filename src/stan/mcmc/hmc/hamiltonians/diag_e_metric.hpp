#ifndef STAN_MCMC_HMC_HAMILTONIANS_DIAG_E_METRIC_HPP
#define STAN_MCMC_HMC_HAMILTONIANS_DIAG_E_METRIC_HPP

#include <stan/callbacks/logger.hpp>
#include <stan/mcmc/hmc/hamiltonians/base_hamiltonian.hpp>
#include <stan/mcmc/hmc/hamiltonians/diag_e_point.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/normal_distribution.hpp>

namespace stan {
namespace mcmc {

// Euclidean manifold with diagonal metric
template <class Model, class BaseRNG>
class diag_e_metric : public base_hamiltonian<diag_e_metric<Model, BaseRNG>, Model, diag_e_point, BaseRNG> {
 public:
  explicit diag_e_metric(const Model& model)
      : base_hamiltonian<diag_e_metric<Model, BaseRNG>, Model, diag_e_point, BaseRNG>(model) {
        dtau_dq_ = Eigen::VectorXd::Zero(this->model_.num_params_r());
  }

  Eigen::VectorXd dtau_dq_;
  inline auto T(diag_e_point& z) {
    return 0.5 * z.p.dot(z.inv_e_metric_.cwiseProduct(z.p));
  }

  inline auto tau(diag_e_point& z) { return T(z); }

  inline auto phi(diag_e_point& z) { return this->V(z); }

  inline auto dG_dt(diag_e_point& z, callbacks::logger& logger) {
    return 2 * T(z) - z.q.dot(z.g);
  }

  inline auto dtau_dq(diag_e_point& z, callbacks::logger& logger) {
    return dtau_dq_;
  }

  inline const auto dtau_dq(diag_e_point& z, callbacks::logger& logger) const {
    return dtau_dq_;
  }

  inline auto dtau_dp(diag_e_point& z) {
    return z.inv_e_metric_.cwiseProduct(z.p);
  }

  inline auto dphi_dq(diag_e_point& z, callbacks::logger& logger) {
    return z.g;
  }

  inline void sample_p(diag_e_point& z, BaseRNG& rng) {
    boost::variate_generator<BaseRNG&, boost::normal_distribution<> >
        rand_diag_gaus(rng, boost::normal_distribution<>());

    for (int i = 0; i < z.p.size(); ++i)
      z.p(i) = rand_diag_gaus() / sqrt(z.inv_e_metric_(i));
  }
};

}  // namespace mcmc
}  // namespace stan
#endif
