#ifndef STAN__AGRAD__REV__FUNCTIONS__FMIN_HPP
#define STAN__AGRAD__REV__FUNCTIONS__FMIN_HPP

#include <stan/agrad/rev/var.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <stan/agrad/rev/internal/precomp_v_vari.hpp>
#include <stan/agrad/rev/internal/precomputed_gradients.hpp>
#include <stan/math/constants.hpp>
#include <stan/meta/likely.hpp>

namespace stan {
  namespace agrad {

    /**
     * Returns the minimum of the two variable arguments (C99).
     *
     * See boost::math::fmin() for the double-based version.
     *
     * For <code>fmin(a,b)</code>, if a's value is less than b's,
     * then a is returned, otherwise b is returned.
     * 
       \f[
       \mbox{fmin}(x,y) = 
       \begin{cases}
         x & \mbox{if } x \leq y \\
         y & \mbox{if } x > y \\[6pt]
         x & \mbox{if } -\infty\leq x\leq \infty, y = \textrm{NaN}\\
         y & \mbox{if } -\infty\leq y\leq \infty, x = \textrm{NaN}\\
         \textrm{NaN} & \mbox{if } x,y = \textrm{NaN}
       \end{cases}
       \f]

       \f[
       \frac{\partial\,\mbox{fmin}(x,y)}{\partial x} = 
       \begin{cases}
         1 & \mbox{if } x \leq y \\
         0 & \mbox{if } x > y \\[6pt]
         1 & \mbox{if } -\infty\leq x\leq \infty, y = \textrm{NaN}\\
         0 & \mbox{if } -\infty\leq y\leq \infty, x = \textrm{NaN}\\
         \textrm{NaN} & \mbox{if } x,y = \textrm{NaN}
       \end{cases}
       \f]

       \f[
       \frac{\partial\,\mbox{fmin}(x,y)}{\partial y} = 
       \begin{cases}
         0 & \mbox{if } x \leq y \\
         1 & \mbox{if } x > y \\[6pt]
         0 & \mbox{if } -\infty\leq x\leq \infty, y = \textrm{NaN}\\
         1 & \mbox{if } -\infty\leq y\leq \infty, x = \textrm{NaN}\\
         \textrm{NaN} & \mbox{if } x,y = \textrm{NaN}
       \end{cases}
       \f]
     *
     * @param a First variable.
     * @param b Second variable.
     * @return If the first variable's value is smaller than the
     * second's, the first variable, otherwise the second variable.
     */
    inline var fmin(const stan::agrad::var& a,
                    const stan::agrad::var& b) {
      if (unlikely(boost::math::isnan(a.vi_->val_))) {
        if(boost::math::isnan(b.vi_->val_)) {
          std::vector<stan::agrad::var> vars;
          std::vector<double> grads;
          vars.push_back(a);
          vars.push_back(b);
          grads.push_back(stan::math::NOT_A_NUMBER);
          grads.push_back(stan::math::NOT_A_NUMBER);
          return var(precomputed_gradients(stan::math::NOT_A_NUMBER,
                                           vars, grads));
        }
        else
          return b;
      } else if (unlikely(boost::math::isnan(b.vi_->val_)))
        return a;
      else
        return a.vi_->val_ < b.vi_->val_ ? a : b;
    }

    /**
     * Returns the minimum of the variable and scalar, promoting the
     * scalar to a variable if it is larger (C99).
     *
     * See boost::math::fmin() for the double-based version.
     * 
     * For <code>fmin(a,b)</code>, if a's value is less than b, then a
     * is returned, otherwise a fresh variable wrapping b is returned.
     * 
     * @param a First variable.
     * @param b Second value
     * @return If the first variable's value is less than or equal to the second value,
     * the first variable, otherwise the second value promoted to a fresh variable.
     */
    inline var fmin(const stan::agrad::var& a,
                    const double& b) {
      if (unlikely(boost::math::isnan(a.vi_->val_))) {
        if(boost::math::isnan(b))
          return var(new precomp_v_vari(stan::math::NOT_A_NUMBER,
                                        a.vi_,
                                        stan::math::NOT_A_NUMBER));
        else
          return var(b);
      } else if (unlikely(boost::math::isnan(b)))
        return a;
      else
        return a.vi_->val_ <= b ? a : var(b);
    }

    /**
     * Returns the minimum of a scalar and variable, promoting the scalar to
     * a variable if it is larger (C99).
     *
     * See boost::math::fmin() for the double-based version.
     * 
     * For <code>fmin(a,b)</code>, if a is less than b's value, then a
     * fresh variable implementation wrapping a is returned, otherwise
     * b is returned.
     *
     * @param a First value.
     * @param b Second variable.
     * @return If the first value is smaller than the second variable's value,
     * return the first value promoted to a variable, otherwise return the 
     * second variable.
     */
    inline var fmin(const double& a,
                    const stan::agrad::var& b) {
      if (unlikely(boost::math::isnan(b.vi_->val_))) {
        if(boost::math::isnan(a))
          return var(new precomp_v_vari(stan::math::NOT_A_NUMBER,
                                        b.vi_,
                                        stan::math::NOT_A_NUMBER));
        else
          return var(a);
      } else if (unlikely(boost::math::isnan(a)))
        return b;
      else
        return a < b.vi_->val_ ? var(a) : b;
    }

  }
}
#endif
