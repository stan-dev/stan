#ifndef __STAN__AGRAD__FWD__HYPOT__HPP__
#define __STAN__AGRAD__FWD__HYPOT__HPP__

#include <stan/agrad/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>
#include <boost/math/special_functions/hypot.hpp>


namespace stan{

  namespace agrad{

    template <typename T1, typename T2>
    inline
    fvar<typename stan::return_type<T1,T2>::type>
    hypot(const fvar<T1>& x1, const fvar<T2>& x2) {
      using boost::math::hypot;
      using std::sqrt;
    return fvar<typename 
                stan::return_type<T1,T2>::type>(hypot(x1.val_, x2.val_), 
                                    (x1.d_ * x1.val_ + x2.d_ * x2.val_) 
                                      / hypot(x1.val_, x2.val_));
    }

    template <typename T1, typename T2>
    inline
    fvar<typename stan::return_type<T1,T2>::type>
    hypot(const fvar<T1>& x1, const T2& x2) {
      using boost::math::hypot;
      using std::sqrt;
    return fvar<typename 
                stan::return_type<T1,T2>::type>(hypot(x1.val_, x2), 
                       (x1.d_ * x1.val_) / hypot(x1.val_, x2));
    }

    template <typename T1, typename T2>
    inline
    fvar<typename stan::return_type<T1,T2>::type>
    hypot(const T1& x1, const fvar<T2>& x2) {
      using boost::math::hypot;
      using std::sqrt;
    return fvar<typename 
                stan::return_type<T1,T2>::type>(hypot(x1, x2.val_), 
                       (x2.d_ * x2.val_) / hypot(x1, x2.val_));
    }

  }
}
#endif
