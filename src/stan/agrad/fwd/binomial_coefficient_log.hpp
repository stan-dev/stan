#ifndef __STAN__AGRAD__FWD__BINOMIAL__COEFFICIENT__LOG__HPP__
#define __STAN__AGRAD__FWD__BINOMIAL__COEFFICIENT__LOG__HPP__

#include <stan/agrad/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>
#include <boost/math/special_functions/digamma.hpp>
#include <stan/math/functions/binomial_coefficient_log.hpp>

namespace stan{

  namespace agrad{

    template <typename T1, typename T2>
    inline
    fvar<typename stan::return_type<T1,T2>::type>
    binomial_coefficient_log(const fvar<T1>& x1, const fvar<T2>& x2){
      using boost::math::digamma;
      using std::log;
      using stan::math::binomial_coefficient_log;
      const double cutoff = 1000;
      if ((x1.val_ < cutoff) || (x1.val_ - x2.val_ < cutoff)) {
        return fvar<typename stan::return_type<T1,T2>::type>(
            binomial_coefficient_log(x1.val_, x2.val_),
                 x1.d_ * digamma(x1.val_ + 1)
               - x2.d_ * digamma(x2.val_ + 1)
               + (x1.d_ - x2.d_) * digamma(x1.val_ - x2.val_ + 1));
      } else {
        return fvar<typename stan::return_type<T1,T2>::type>(
            binomial_coefficient_log(x1.val_, x2.val_), 
               x2.d_ * log(x1.val_ - x2.val_) 
            + x2.val_ * (x1.d_ - x2.d_) / (x1.val_ - x2.val_) 
            + x1.d_ * log(x1.val_ / (x1.val_ - x2.val_))
            + (x1.val_ + 0.5) / (x1.val_ / (x1.val_ - x2.val_))
              * (x1.d_ * (x1.val_ - x2.val_) - (x1.d_ - x2.d_) * x1.val_)
                / ((x1.val_ - x2.val_) * (x1.val_ - x2.val_))
            - x1.d_ / (12.0 * x1.val_ * x1.val_)
            - x2.d_ 
            + (x1.d_ - x2.d_) / (12.0 * (x1.val_ - x2.val_) 
              * (x1.val_ - x2.val_)) 
            - digamma(x2.val_ + 1) * x2.d_);
      }
    }

    template <typename T1, typename T2>
    inline
    fvar<typename stan::return_type<T1,T2>::type>
    binomial_coefficient_log(const fvar<T1>& x1, const T2& x2){
      using boost::math::digamma;
      using std::log;
      using stan::math::binomial_coefficient_log;
      const double cutoff = 1000;
      if ((x1.val_ < cutoff) || (x1.val_ - x2 < cutoff)) {
        return fvar<typename stan::return_type<T1,T2>::type>(
                        binomial_coefficient_log(x1.val_, x2),
                          x1.d_ * digamma(x1.val_ + 1)
                        + x1.d_ * digamma(x1.val_ - x2 + 1));
      } else {
        return fvar<typename stan::return_type<T1,T2>::type>( 
            binomial_coefficient_log(x1.val_, x2), 
              x2 * x1.d_ / (x1.val_ - x2) 
            + x1.d_ * log(x1.val_ / (x1.val_ - x2))
            + (x1.val_ + 0.5) / (x1.val_ / (x1.val_ - x2))
              * (x1.d_ * (x1.val_ - x2) - x1.d_ * x1.val_)
                / ((x1.val_ - x2) * (x1.val_ - x2))
            - x1.d_ / (12.0 * x1.val_ * x1.val_)
            + x1.d_ / (12.0 * (x1.val_ - x2) * (x1.val_ - x2)));
      }
    }

    template <typename T1, typename T2>
    inline
    fvar<typename stan::return_type<T1,T2>::type>
    binomial_coefficient_log(const T1& x1, const fvar<T2>& x2){
      using boost::math::digamma;
      using std::log;
      using stan::math::binomial_coefficient_log;
      const double cutoff = 1000;
      if ((x1 < cutoff) || (x1 - x2.val_ < cutoff)) {
        return fvar<typename stan::return_type<T1,T2>::type>(
                   binomial_coefficient_log(x1, x2.val_),
                   - x2.d_ * digamma(x2.val_ + 1) 
                   - x2.d_ * digamma(x1 - x2.val_ + 1));
      } else {
        return fvar<typename stan::return_type<T1,T2>::type>(
            binomial_coefficient_log(x1, x2.val_), 
               x2.d_ * log(x1 - x2.val_) 
            + x2.val_ * -x2.d_ / (x1 - x2.val_) 
            - x2.d_ 
            - x2.d_ / (12.0 * (x1 - x2.val_) * (x1 - x2.val_)) 
            - digamma(x2.val_ + 1) * x2.d_);
      }
    }
  }
}
#endif
