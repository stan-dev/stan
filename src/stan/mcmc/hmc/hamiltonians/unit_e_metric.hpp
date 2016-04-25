#ifndef STAN_MCMC_HMC_HAMILTONIANS_UNIT_E_METRIC_HPP
#define STAN_MCMC_HMC_HAMILTONIANS_UNIT_E_METRIC_HPP

#include <stan/mcmc/hmc/hamiltonians/base_hamiltonian.hpp>
#include <stan/mcmc/hmc/hamiltonians/unit_e_point.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/normal_distribution.hpp>

namespace stan {
  namespace mcmc {

    // Euclidean manifold with unit metric
    template <class Model, class BaseRNG>
    class unit_e_metric
      : public base_hamiltonian<Model, unit_e_point, BaseRNG> {
    public:
      explicit unit_e_metric(const Model& model)
        : base_hamiltonian<Model, unit_e_point, BaseRNG>(model) {}

      ~unit_e_metric() {}

      double T(unit_e_point& z) {
        return 0.5 * z.p.squaredNorm();
      }

      double tau(unit_e_point& z) {
        return T(z);
      }

      double phi(unit_e_point& z) {
        return this->V(z);
      }

      double dG_dt(unit_e_point& z) {
        return 2 * T(z) - z.q.dot(z.g);
      }

      const Eigen::VectorXd dtau_dq(unit_e_point& z) {
        return Eigen::VectorXd::Zero(this->model_.num_params_r());
      }

      const Eigen::VectorXd dtau_dp(unit_e_point& z) {
        return z.p;
      }

      const Eigen::VectorXd dphi_dq(unit_e_point& z) {
        return z.g;
      }

      void sample_p(unit_e_point& z, BaseRNG& rng) {
        boost::variate_generator<BaseRNG&, boost::normal_distribution<> >
          rand_unit_gaus(rng, boost::normal_distribution<>());

        for (int i = 0; i < z.p.size(); ++i)
          z.p(i) = rand_unit_gaus();
      }

      double stein(unit_e_point& z, double alpha, double beta) {
        double quad = 0;
        for (int n = 0; n < z.q.size(); ++n)
          quad += z.q(n) * z.q(n);

        double quad_pow = std::pow(quad, 0.5 * beta);
        double u = std::exp(-alpha * quad_pow);

        double stein = alpha * beta * std::pow(quad, 0.5 * beta - 1) * u;
        stein *= z.q.dot(z.g) - z.q.size() - beta + 2
                 + alpha * beta * quad_pow;

        return stein;
      }

      double score(unit_e_point& z) {
        double sum = 0;
        for (int n = 0; n < z.q.size(); ++n)
          sum += z.g(n);
        return sum / z.q.size();
      }
    };

  }  // mcmc
}  // stan
#endif
