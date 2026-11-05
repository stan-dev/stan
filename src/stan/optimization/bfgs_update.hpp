#ifndef STAN_OPTIMIZATION_BFGS_UPDATE_HPP
#define STAN_OPTIMIZATION_BFGS_UPDATE_HPP

#include <stan/math/prim/fun/Eigen.hpp>

namespace stan {
namespace optimization {
template <typename Scalar = double, int DimAtCompile = Eigen::Dynamic>
class BFGSUpdate_HInv {
 public:
  typedef Eigen::Matrix<Scalar, DimAtCompile, 1> VectorT;
  typedef Eigen::Matrix<Scalar, DimAtCompile, DimAtCompile> HessianT;

  /**
   * Update the inverse Hessian approximation.
   *
   * @param yk Difference between the current and previous gradient vector.
   * @param sk Difference between the current and previous state vector.
   * @param reset Whether to reset the approximation, forgetting about
   * previous values.
   * @return In the case of a reset, returns the optimal scaling of the
   * initial Hessian approximation which is useful for predicting
   * step-sizes.
   **/
  inline Scalar update(const VectorT &yk, const VectorT &sk,
                       bool reset = false) {
    Scalar rhok, skyk, B0fact;

    skyk = yk.dot(sk);
    rhok = 1.0 / skyk;

    if (reset) {
      B0fact = yk.squaredNorm() / skyk;
      _Hk.setIdentity(yk.size(), yk.size());
      _Hk /= B0fact;
    } else {
      B0fact = 1.0;
    }
    // H <- (I - rho s y') H (I - rho y s') + rho s s'
    //    = H + s w' + w s',  u = H y,  w = (c / 2) s - rho u,
    //    c = rho^2 y'u + rho
    // Only the lower triangle of _Hk is kept up to date.
    _u.noalias() = _Hk.template selfadjointView<Eigen::Lower>() * yk;
    const Scalar c = rhok * rhok * yk.dot(_u) + rhok;
    _w = (0.5 * c) * sk - rhok * _u;
    // rankUpdate copies plain vector arguments; Maps avoid the allocations
    _Hk.template selfadjointView<Eigen::Lower>().rankUpdate(
        Eigen::Map<const VectorT>(sk.data(), sk.size()),
        Eigen::Map<const VectorT>(_w.data(), _w.size()));
    return B0fact;
  }

  /**
   * Compute the search direction based on the current (inverse) Hessian
   * approximation and given gradient.
   *
   * @param[out] pk The negative product of the inverse Hessian and gradient
   * direction gk.
   * @param[in] gk Gradient direction.
   **/
  inline void search_direction(VectorT &pk, const VectorT &gk) const {
    pk.noalias() = _Hk.template selfadjointView<Eigen::Lower>() * (-gk);
  }

 private:
  HessianT _Hk;
  VectorT _u, _w;
};
}  // namespace optimization
}  // namespace stan

#endif
