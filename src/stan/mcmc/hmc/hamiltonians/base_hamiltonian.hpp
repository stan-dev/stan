#ifndef STAN_MCMC_HMC_HAMILTONIANS_BASE_HAMILTONIAN_HPP
#define STAN_MCMC_HMC_HAMILTONIANS_BASE_HAMILTONIAN_HPP

#include <stan/callbacks/logger.hpp>
#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/model/gradient.hpp>
#include <stan/model/log_prob_propto.hpp>
#include <Eigen/Dense>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace stan {
namespace mcmc {

/**
 * Base CRTP class for hamiltonians.
 * @tparam Derived type that inherits from base_hamiltonian and defines members
 * functions @c T, @c tau @c phi @c dG_dt, @c dtau_dq, @c dtau_dp, @c dphi_dq,
 * and @c sample_p
 * @tparam Model class that can take returns log probability
 * @tparam PointType type that inherits from ps_point
 * @tparam BaseRNG a random number generator class.
 */
template <typename Derived, typename Model, typename PointType,
          typename BaseRNG>
class base_hamiltonian {
 public:
  explicit base_hamiltonian(const Model& model) : model_(model) {}

  using point_type = PointType;
  // modifier to the derived class
  inline Derived& derived() { return static_cast<Derived&>(*this); }
  // inspector to the derived class
  inline const Derived& derived() const {
    return static_cast<Derived const&>(*this);
  }

  inline auto T(point_type& z) { return this->derived().T(z); };

  inline auto V(point_type& z) { return z.V; }

  inline auto tau(point_type& z) { return this->derived().tau(z); };

  inline auto phi(point_type& z) { return this->derived().phi(z); };

  inline auto H(point_type& z) { return T(z) + V(z); }

  // The time derivative of the virial, G = \sum_{d = 1}^{D} q^{d} p_{d}.
  inline auto dG_dt(point_type& z, callbacks::logger& logger) {
    return this->derived().dG_dt(z, logger);
  };

  // tau = 0.5 p_{i} p_{j} Lambda^{ij} (q)
  inline auto dtau_dq(point_type& z, callbacks::logger& logger) {
    return this->derived().dtau_dq(z, logger);
  };

  inline auto dtau_dp(point_type& z) { return this->derived().dtau_dp(z); };

  // phi = 0.5 * log | Lambda (q) | + V(q)
  inline auto dphi_dq(point_type& z, callbacks::logger& logger) {
    return this->derived().dphi_dq(z, logger);
  };

  inline void sample_p(point_type& z, BaseRNG& rng) {
    this->derived().sample_p(z, rng);
  };

  inline void init(point_type& z, callbacks::logger& logger) {
    this->update_potential_gradient(z, logger);
  }

  inline void update_potential(point_type& z, callbacks::logger& logger) {
    try {
      z.V = -stan::model::log_prob_propto<true>(model_, z.q);
    } catch (const std::exception& e) {
      this->write_error_msg_(e, logger);
      z.V = std::numeric_limits<double>::infinity();
    }
  }

  inline void update_potential_gradient(point_type& z,
                                        callbacks::logger& logger) {
    try {
      stan::model::gradient(model_, z.q, z.V, z.g, logger);
      z.V = -z.V;
    } catch (const std::exception& e) {
      this->write_error_msg_(e, logger);
      z.V = std::numeric_limits<double>::infinity();
    }
    z.g = -z.g;
  }

  inline void update_metric(point_type& z, callbacks::logger& logger) {}

  inline void update_metric_gradient(point_type& z, callbacks::logger& logger) {
  }

  inline void update_gradients(point_type& z, callbacks::logger& logger) {
    update_potential_gradient(z, logger);
  }

 protected:
  const Model& model_;

  inline void write_error_msg_(const std::exception& e,
                               callbacks::logger& logger) {
    logger.error(
        "Informational Message: The current Metropolis proposal "
        "is about to be rejected because of the following issue:");
    logger.error(e.what());
    logger.error(
        "If this warning occurs sporadically, such as for highly "
        "constrained variable types like covariance matrices, "
        "then the sampler is fine,");
    logger.error(
        "but if this warning occurs often then your model may be "
        "either severely ill-conditioned or misspecified.");
    logger.error("");
  }
};

}  // namespace mcmc
}  // namespace stan

#endif
