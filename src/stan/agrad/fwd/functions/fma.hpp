#ifndef STAN__AGRAD__FWD__FUNCTIONS__FMA_HPP
#define STAN__AGRAD__FWD__FUNCTIONS__FMA_HPP

#include <cmath>
#include <stan/agrad/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>

namespace stan {

  namespace agrad {

    /**
     * The fused multiply-add operation (C99).   
     *
     * This double-based operation delegates to <code>fma</code>.
     *
     * The function is defined by
     *
     * <code>fma(a,b,c) = (a * b) + c</code>.
     *
     *
       \f[
       \mbox{fma}(x,y,z) = 
       \begin{cases}
         x\cdot y+z & \mbox{if } -\infty\leq x,y,z \leq \infty \\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN}
       \end{cases}
       \f]
       
       \f[
       \frac{\partial\,\mbox{fma}(x,y,z)}{\partial x} = 
       \begin{cases}
         y & \mbox{if } -\infty\leq x,y,z \leq \infty \\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN}
       \end{cases}
       \f]
       
       \f[
       \frac{\partial\,\mbox{fma}(x,y,z)}{\partial y} = 
       \begin{cases}
         x & \mbox{if } -\infty\leq x,y,z \leq \infty \\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN}
       \end{cases}
       \f]
       
       \f[
       \frac{\partial\,\mbox{fma}(x,y,z)}{\partial z} = 
       \begin{cases}
         1 & \mbox{if } -\infty\leq x,y,z \leq \infty \\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN}
       \end{cases}
       \f]
     *
     * @param x1 First value.
     * @param x2 Second value.
     * @param x3 Third value.
     * @return Product of the first two values plus the third.
     */
    template <typename T1, typename T2, typename T3>
    inline
    fvar<typename stan::return_type<T1,T2,T3>::type>
    fma(const fvar<T1>& x1, const fvar<T2>& x2, const fvar<T3>& x3) {
      using ::fma;
      return fvar<typename stan::return_type<T1,T2,T3>::type>
        (fma(x1.val_, x2.val_, x3.val_), x1.d_ * x2.val_ + x2.d_ * x1.val_ + x3.d_);
    }

    /**
     * See all-var input signature for details on the function and derivatives.
     */
    template <typename T1, typename T2, typename T3>
    inline
    fvar<typename stan::return_type<T1,T2,T3>::type>
    fma(const T1& x1, const fvar<T2>& x2, const fvar<T3>& x3) {
      using ::fma;
      return fvar<typename stan::return_type<T1,T2,T3>::type>
        (fma(x1, x2.val_, x3.val_), x2.d_ * x1 + x3.d_);
    }

    /**
     * See all-var input signature for details on the function and derivatives.
     */
    template <typename T1, typename T2, typename T3>
    inline
    fvar<typename stan::return_type<T1,T2,T3>::type>
    fma(const fvar<T1>& x1, const T2& x2, const fvar<T3>& x3) {
      using ::fma;
      return fvar<typename stan::return_type<T1,T2,T3>::type>
        (fma(x1.val_, x2, x3.val_), x1.d_ * x2 + x3.d_);
    }

    /**
     * See all-var input signature for details on the function and derivatives.
     */
    template <typename T1, typename T2, typename T3>
    inline
    fvar<typename stan::return_type<T1,T2,T3>::type>
    fma(const fvar<T1>& x1, const fvar<T2>& x2, const T3& x3) {
      using ::fma;
      return fvar<typename stan::return_type<T1,T2,T3>::type>
        (fma(x1.val_, x2.val_, x3), x1.d_ * x2.val_ + x2.d_ * x1.val_);
    }

    /**
     * See all-var input signature for details on the function and derivatives.
     */
    template <typename T1, typename T2, typename T3>
    inline
    fvar<typename stan::return_type<T1,T2,T3>::type>
    fma(const T1& x1, const T2& x2, const fvar<T3>& x3) {
      using ::fma;
      return fvar<typename stan::return_type<T1,T2,T3>::type>
        (fma(x1, x2, x3.val_), x3.d_);
    }

    /**
     * See all-var input signature for details on the function and derivatives.
     */
    template <typename T1, typename T2, typename T3>
    inline
    fvar<typename stan::return_type<T1,T2,T3>::type>
    fma(const fvar<T1>& x1, const T2& x2, const T3& x3) {
      using ::fma;
      return fvar<typename stan::return_type<T1,T2,T3>::type>
        (fma(x1.val_, x2, x3), x1.d_ * x2);
    }

    /**
     * See all-var input signature for details on the function and derivatives.
     */
    template <typename T1, typename T2, typename T3>
    inline
    fvar<typename stan::return_type<T1,T2,T3>::type>
    fma(const T1& x1, const fvar<T2>& x2, const T3& x3) {
      using ::fma;
      return fvar<typename stan::return_type<T1,T2,T3>::type>
        (fma(x1, x2.val_, x3), x2.d_ * x1);
    }

  }
}
#endif
