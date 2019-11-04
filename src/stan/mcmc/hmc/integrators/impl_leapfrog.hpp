#ifndef STAN_MCMC_HMC_INTEGRATORS_IMPL_LEAPFROG_HPP
#define STAN_MCMC_HMC_INTEGRATORS_IMPL_LEAPFROG_HPP

#include <Eigen/Dense>
#include <stan/mcmc/hmc/integrators/base_leapfrog.hpp>

namespace stan {
namespace mcmc {

template <typename Hamiltonian>
class impl_leapfrog : public base_leapfrog<Hamiltonian> {
 public:
  impl_leapfrog()
      : base_leapfrog<Hamiltonian>(),
        max_num_fixed_point_(10),
        fixed_point_threshold_(1e-8) {}
  using point_type = typename Hamiltonian::PointType;

  inline void begin_update_p(point_type& z, Hamiltonian& hamiltonian,
                             double epsilon, callbacks::logger& logger) {
    hat_phi(z, hamiltonian, epsilon, logger);
    hat_tau(z, hamiltonian, epsilon, this->max_num_fixed_point_, logger);
  }

  inline void update_q(point_type& z, Hamiltonian& hamiltonian, double epsilon,
                       callbacks::logger& logger) {
    // hat{T} = dT/dp * d/dq
    Eigen::VectorXd q_init = z.q + 0.5 * epsilon * hamiltonian.dtau_dp(z);
    Eigen::VectorXd delta_q(z.q.size());

    for (int n = 0; n < this->max_num_fixed_point_; ++n) {
      delta_q = z.q;
      z.q.noalias() = q_init + 0.5 * epsilon * hamiltonian.dtau_dp(z);
      hamiltonian.update_metric(z, logger);

      delta_q -= z.q;
      if (delta_q.cwiseAbs().maxCoeff() < this->fixed_point_threshold_)
        break;
    }
    hamiltonian.update_gradients(z, logger);
  }

  inline void end_update_p(point_type& z, Hamiltonian& hamiltonian,
                           double epsilon, callbacks::logger& logger) {
    hat_tau(z, hamiltonian, epsilon, 1, logger);
    hat_phi(z, hamiltonian, epsilon, logger);
  }

  // hat{phi} = dphi/dq * d/dp
  inline void hat_phi(point_type& z, Hamiltonian& hamiltonian, double epsilon,
                      callbacks::logger& logger) {
    z.p -= epsilon * hamiltonian.dphi_dq(z, logger);
  }

  // hat{tau} = dtau/dq * d/dp
  inline void hat_tau(point_type& z, Hamiltonian& hamiltonian, double epsilon,
                      int num_fixed_point, callbacks::logger& logger) {
    Eigen::VectorXd p_init = z.p;
    Eigen::VectorXd delta_p(z.p.size());

    for (int n = 0; n < num_fixed_point; ++n) {
      delta_p = z.p;
      z.p.noalias() = p_init - epsilon * hamiltonian.dtau_dq(z, logger);
      delta_p -= z.p;
      if (delta_p.cwiseAbs().maxCoeff() < this->fixed_point_threshold_)
        break;
    }
  }

  inline int& max_num_fixed_point() { return this->max_num_fixed_point_; }
  inline const int& max_num_fixed_point() const {
    return this->max_num_fixed_point_;
  }

  inline double& fixed_point_threshold() {
    return this->fixed_point_threshold_;
  }
  inline double& const fixed_point_threshold() const {
    return this->fixed_point_threshold_;
  }

 private:
  int max_num_fixed_point_;
  double fixed_point_threshold_;
};

}  // namespace mcmc
}  // namespace stan

#endif
