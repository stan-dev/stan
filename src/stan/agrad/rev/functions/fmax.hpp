#ifndef STAN__AGRAD__REV__FUNCTIONS__FMAX_HPP
#define STAN__AGRAD__REV__FUNCTIONS__FMAX_HPP

#include <stan/agrad/rev/var.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <stan/agrad/rev/internal/precomp_v_vari.hpp>
#include <stan/agrad/rev/internal/precomputed_gradients.hpp>
#include <stan/math/constants.hpp>
#include <stan/meta/likely.hpp>

namespace stan {
  namespace agrad {

    /**
     * Returns the maximum of the two variable arguments (C99).
     *
     * See boost::math::fmax() for the double-based version.
     * 
     * No new variable implementations are created, with this function
     * defined as if by
     *
     * <code>fmax(a,b) = a</code> if a's value is greater than b's, and .
     *
     * <code>fmax(a,b) = b</code> if b's value is greater than or equal to a's.
     * 
       \f[
       \mbox{fmax}(x,y) = 
       \begin{cases}
         x & \mbox{if } x \geq y \\
         y & \mbox{if } x < y \\[6pt]
         x & \mbox{if } -\infty\leq x\leq \infty, y = \textrm{NaN}\\
         y & \mbox{if } -\infty\leq y\leq \infty, x = \textrm{NaN}\\
         \textrm{NaN} & \mbox{if } x,y = \textrm{NaN}
       \end{cases}
       \f]
       
       \f[
       \frac{\partial\,\mbox{fmax}(x,y)}{\partial x} = 
       \begin{cases}
         1 & \mbox{if } x \geq y \\
         0 & \mbox{if } x < y \\[6pt]
         1 & \mbox{if } -\infty\leq x\leq \infty, y = \textrm{NaN}\\
         0 & \mbox{if } -\infty\leq y\leq \infty, x = \textrm{NaN}\\
         \textrm{NaN} & \mbox{if } x,y = \textrm{NaN}
       \end{cases}
       \f]
       
       \f[
       \frac{\partial\,\mbox{fmax}(x,y)}{\partial y} = 
       \begin{cases}
         0 & \mbox{if } x \geq y \\
         1 & \mbox{if } x < y \\[6pt]
         0 & \mbox{if } -\infty\leq x\leq \infty, y = \textrm{NaN}\\
         1 & \mbox{if } -\infty\leq y\leq \infty, x = \textrm{NaN}\\
         \textrm{NaN} & \mbox{if } x,y = \textrm{NaN}
       \end{cases}
       \f]
     *
     * @param a First variable.
     * @param b Second variable.
     * @return If the first variable's value is larger than the
     * second's, the first variable, otherwise the second variable.
     */
    inline var fmax(const stan::agrad::var& a,
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
        return a.vi_->val_ > b.vi_->val_ ? a : b;
    }

    /**
     * Returns the maximum of the variable and scalar, promoting the
     * scalar to a variable if it is larger (C99).
     *
     * See boost::math::fmax() for the double-based version.
     * 
     * For <code>fmax(a,b)</code>, if a's value is greater than b,
     * then a is returned, otherwise a fesh variable implementation
     * wrapping the value b is returned.
     *
     * @param a First variable.
     * @param b Second value
     * @return If the first variable's value is larger than or equal
     * to the second value, the first variable, otherwise the second
     * value promoted to a fresh variable.
     */
    inline var fmax(const stan::agrad::var& a,
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
        return a.vi_->val_ >= b ? a : var(b);
    }

    /**
     * Returns the maximum of a scalar and variable, promoting the scalar to
     * a variable if it is larger (C99).
     *
     * See boost::math::fmax() for the double-based version.
     * 
     * For <code>fmax(a,b)</code>, if a is greater than b's value,
     * then a fresh variable implementation wrapping a is returned, otherwise 
     * b is returned.
     *
     * @param a First value.
     * @param b Second variable.
     * @return If the first value is larger than the second variable's value,
     * return the first value promoted to a variable, otherwise return the 
     * second variable.
     */
    inline var fmax(const double& a,
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
        return a > b.vi_->val_ ? var(a) : b;
    }

  }
}
#endif
