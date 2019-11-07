#ifndef STAN_MCMC_HMC_HAMILTONIANS_DIAG_E_METRIC_HPP
#define STAN_MCMC_HMC_HAMILTONIANS_DIAG_E_METRIC_HPP

#include <stan/callbacks/logger.hpp>
#include <stan/mcmc/hmc/hamiltonians/base_hamiltonian.hpp>
#include <stan/mcmc/hmc/hamiltonians/diag_e_point.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/normal_distribution.hpp>

#include <iostream>

namespace stan {
namespace mcmc {

// Euclidean manifold with diagonal metric
template <class Model, class BaseRNG>
class diag_e_metric : public base_hamiltonian<Model, diag_e_point, BaseRNG> {
 public:
  explicit diag_e_metric(const Model& model)
      : base_hamiltonian<Model, diag_e_point, BaseRNG>(model) {}

  double T(diag_e_point& z) {
    return 0.5 * z.p.dot(z.inv_e_metric_.cwiseProduct(z.p));
  }

  double tau(diag_e_point& z) { return T(z); }

  double phi(diag_e_point& z) { return this->V(z); }

  double dG_dt(diag_e_point& z, callbacks::logger& logger) {
    return 2 * T(z) - z.q.dot(z.g);
  }

  Eigen::VectorXd dtau_dq(diag_e_point& z, callbacks::logger& logger) {
    return Eigen::VectorXd::Zero(this->model_.num_params_r());
  }

  Eigen::VectorXd dtau_dp(diag_e_point& z) {
    return z.inv_e_metric_.cwiseProduct(z.p);
  }

  Eigen::VectorXd dphi_dq(diag_e_point& z, callbacks::logger& logger) {
    return z.g;
  }

  void sample_p(diag_e_point& z, BaseRNG& rng) {
    boost::random::variate_generator<BaseRNG&, boost::random::normal_distribution<> >
        rand_diag_gaus(rng, boost::random::normal_distribution<>());
    //boost::random::normal_distribution<> std_normal(0.0, 1.0);

    //std::cout << "sampling moments " << z.p.size() << std::endl;
    for (int i = 0; i < z.p.size(); ++i) {
      //std::cout << "component " << i << " scaled by " << sqrt(z.inv_e_metric_(i)) << std::endl;
      //const double draw = std_normal(rng);
      //std::cout << "normal variate " << draw << std::endl;
      z.p(i) = rand_diag_gaus() / sqrt(z.inv_e_metric_(i));
      //z.p(i) = draw / sqrt(z.inv_e_metric_(i));
      //std::cout << "got " << z.p(i) << std::endl;
    }
    //std::cout << "sampling moments done" << std::endl;
  }
};

}  // namespace mcmc
}  // namespace stan
#endif
