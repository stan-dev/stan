#ifndef STAN_MCMC_HMC_INTEGRATORS_BASE_LEAPFROG_HPP
#define STAN_MCMC_HMC_INTEGRATORS_BASE_LEAPFROG_HPP

#include <stan/callbacks/logger.hpp>
#include <stan/mcmc/hmc/integrators/base_integrator.hpp>
#include <iostream>
#include <iomanip>

namespace stan {
namespace mcmc {

/**
 * Base class for leapfrog integrators.
 * @tparam Derived class that defines methods /c begin_update_p,
 *  /c end_update_p, /c update_q.
 * @tparam Hamiltonian class representing a hamiltonian.
 */
template <typename Derived, typename Hamiltonian>
class base_leapfrog
    : public base_integrator<base_leapfrog<Derived, Hamiltonian>, Hamiltonian> {
 public:
  base_leapfrog()
      : base_integrator<base_leapfrog<Derived, Hamiltonian>, Hamiltonian>() {}

  using hamiltonian_type = Hamiltonian;
  using point_type = typename Hamiltonian::point_type;
  // modifier to the derived class
  inline Derived& derived() { return static_cast<Derived&>(*this); }
  // inspector to the derived class
  inline const Derived& derived() const {
    return static_cast<Derived const&>(*this);
  }


  inline void verbose_evolve(point_type& z,
                      Hamiltonian& hamiltonian, const double epsilon,
                      callbacks::logger& logger) {
    std::stringstream msg;
    msg.precision(6);

    int width = 14;
    int nColumn = 4;

    msg << "Verbose Hamiltonian Evolution, Step Size = " << epsilon << ":";
    logger.info(msg);

    msg.str("");
    msg << "    " << std::setw(nColumn * width) << std::setfill('-') << ""
        << std::setfill(' ');
    logger.info(msg);

    msg.str("");
    msg << "    " << std::setw(width) << std::left << "Poisson"
        << std::setw(width) << std::left << "Initial" << std::setw(width)
        << std::left << "Current" << std::setw(width) << std::left << "DeltaH";
    logger.info(msg);

    msg.str("");
    msg << "    " << std::setw(width) << std::left << "Operator"
        << std::setw(width) << std::left << "Hamiltonian" << std::setw(width)
        << std::left << "Hamiltonian" << std::setw(width) << std::left
        << "/ Stepsize^{2}";
    logger.info(msg);

    msg.str("");
    msg << "    " << std::setw(nColumn * width) << std::setfill('-') << ""
        << std::setfill(' ');
    logger.info(msg);

    double H0 = hamiltonian.H(z);

    begin_update_p(z, hamiltonian, 0.5 * epsilon, logger);

    double H1 = hamiltonian.H(z);

    msg.str("");
    msg << "    " << std::setw(width) << std::left << "hat{V}/2"
        << std::setw(width) << std::left << H0 << std::setw(width) << std::left
        << H1 << std::setw(width) << std::left
        << (H1 - H0) / (epsilon * epsilon);
    logger.info(msg);

    update_q(z, hamiltonian, epsilon, logger);

    double H2 = hamiltonian.H(z);

    msg.str("");
    msg << "    " << std::setw(width) << std::left << "hat{T}"
        << std::setw(width) << std::left << H0 << std::setw(width) << std::left
        << H2 << std::setw(width) << std::left
        << (H2 - H0) / (epsilon * epsilon);
    logger.info(msg);

    end_update_p(z, hamiltonian, 0.5 * epsilon, logger);

    double H3 = hamiltonian.H(z);

    msg.str("");
    msg << "    " << std::setw(width) << std::left << "hat{V}/2"
        << std::setw(width) << std::left << H0 << std::setw(width) << std::left
        << H3 << std::setw(width) << std::left
        << (H3 - H0) / (epsilon * epsilon);
    logger.info(msg);

    msg.str("");
    msg << "    " << std::setw(nColumn * width) << std::setfill('-') << ""
        << std::setfill(' ');
    logger.info(msg);
  }

  inline void evolve(point_type& z, hamiltonian_type& hamiltonian,
                     const double epsilon, callbacks::logger& logger) {
    this->derived().begin_update_p(z, hamiltonian, 0.5 * epsilon, logger);
    this->derived().update_q(z, hamiltonian, epsilon, logger);
    this->derived().end_update_p(z, hamiltonian, 0.5 * epsilon, logger);
  }

  inline void begin_update_p(point_type& z, hamiltonian_type& hamiltonian,
                             double epsilon, callbacks::logger& logger) {
    this->derived().begin_update_p(z, hamiltonian, epsilon, logger);
  }

  inline void update_q(point_type& z, hamiltonian_type& hamiltonian,
                       double epsilon, callbacks::logger& logger) {
    this->derived().update_q(z, hamiltonian, epsilon, logger);
  }
  inline void end_update_p(point_type& z, hamiltonian_type& hamiltonian,
                           double epsilon, callbacks::logger& logger) {
    this->derived().end_update_p(z, hamiltonian, epsilon, logger);
  }
};

}  // namespace mcmc
}  // namespace stan
#endif
