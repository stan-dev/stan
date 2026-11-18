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
   * Except at a reset, the approximation H is multiplied by
   * tau = s'y / y'Hy before the update if tau > 1 (Oren-Luenberger sizing
   * restricted to enlarging H). Without it H keeps the scale set at the
   * last reset, which far from the optimum can underestimate the inverse
   * curvature by orders of magnitude in the directions not explored since.
   * Scaling all of H can overestimate it along earlier steps instead; a
   * later pair along such a direction has y'Hy > s'y, so it is not sized
   * and the update corrects H there.
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
    //    = H + (c s - rho u) s' - rho s u',  u = H y,  c = rho^2 y'u + rho
    _u.noalias() = _Hk * yk;
    Scalar yu = yk.dot(_u);
    if (!reset && 0 < yu && yu < skyk) {
      const Scalar tau = skyk / yu;
      _Hk *= tau;
      _u *= tau;
      yu = skyk;
    }
    const Scalar c = rhok * rhok * yu + rhok;
    _w = c * sk - rhok * _u;
    _Hk.noalias() += _w * sk.transpose();
    _u *= rhok;
    _Hk.noalias() -= sk * _u.transpose();
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
    pk.noalias() = -(_Hk * gk);
  }

 private:
  HessianT _Hk;
  VectorT _u, _w;
};
}  // namespace optimization
}  // namespace stan

#endif
