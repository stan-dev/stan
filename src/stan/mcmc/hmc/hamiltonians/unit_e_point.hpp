#ifndef STAN_MCMC_HMC_HAMILTONIANS_UNIT_E_POINT_HPP
#define STAN_MCMC_HMC_HAMILTONIANS_UNIT_E_POINT_HPP

#include <stan/mcmc/hmc/hamiltonians/ps_point.hpp>

namespace stan {
namespace mcmc {
/**
 * Point in a phase space with a base
 * Euclidean manifold with unit metric
 */
class unit_e_point : public ps_point {
 public:
  explicit unit_e_point(int n) : ps_point(n) {}
  /**
   * Assign the base @c ps_point class values to this class.
   * @tparam Base A @c ps_point type
   * @param other The @c ps_point whose members @c g @c p @c q will be assigned
   *  to this object.
   */
  template <typename Base, require_same_t<ps_point, Base>...>
  auto& operator=(Base&& other) {
    this->ps_point::operator=(std::forward<Base>(other));
    return *this;
  }

};

inline void write_metric(stan::callbacks::writer& writer) {
  writer("No free parameters for unit metric");
}

}  // namespace mcmc
}  // namespace stan

#endif
