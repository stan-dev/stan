#ifndef STAN__AGRAD__FWD__FUNCTIONS__OWENS_T_HPP
#define STAN__AGRAD__FWD__FUNCTIONS__OWENS_T_HPP

#include <stan/agrad/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>
#include <stan/math/constants.hpp>
#include <stan/math/functions/owens_t.hpp>

namespace stan {
  namespace agrad {

    template <typename T>
    inline
    fvar<T>    
    owens_t(const fvar<T>& x1, const fvar<T>& x2) {
      using stan::math::owens_t;
      using boost::math::erf;
      using stan::math::pi;
      using std::exp;

      T neg_x1_sq_div_2 = -x1.val_ * x1.val_ / 2.0;
      T one_p_x2_sq = 1.0 + x2.val_ * x2.val_;

      return fvar<T>(owens_t(x1.val_, x2.val_), 
                     - x1.d_ 
                     * (erf(x2.val_ * x1.val_ / std::sqrt(2.0)) 
                        * exp(neg_x1_sq_div_2) / std::sqrt(8.0 * pi())) 
                     + x2.d_ * exp(neg_x1_sq_div_2 * one_p_x2_sq) 
                     / (one_p_x2_sq * 2.0 * pi()));
    }

    template <typename T>
    inline
    fvar<T>    
    owens_t(const double x1, const fvar<T>& x2) {
      using stan::math::owens_t;
      using boost::math::erf;
      using stan::math::pi;
      using std::exp;

      T neg_x1_sq_div_2 = -x1 * x1 / 2.0;
      T one_p_x2_sq = 1.0 + x2.val_ * x2.val_;

      return fvar<T>(owens_t(x1, x2.val_), 
                     x2.d_ * exp(neg_x1_sq_div_2 * one_p_x2_sq) 
                     / (one_p_x2_sq * 2.0 * pi()));
    }

    template <typename T>
    inline
    fvar<T>    
    owens_t(const fvar<T>& x1, const double x2) {
      using stan::math::owens_t;
      using boost::math::erf;
      using stan::math::pi;
      using std::exp;

      T neg_x1_sq_div_2 = -x1.val_ * x1.val_ / 2.0;

      return fvar<T>(owens_t(x1.val_, x2), 
                     -x1.d_  * (erf(x2 * x1.val_ / std::sqrt(2.0)) 
                                * exp(neg_x1_sq_div_2) 
                                / std::sqrt(8.0 * pi())));
    }

  }
}
#endif
