#ifndef __STAN__AGRAD__AGRAD_SPECIAL_FUNCTIONS_HPP__
#define __STAN__AGRAD__AGRAD_SPECIAL_FUNCTIONS_HPP__

#include <stan/agrad/boost_fpclassify.hpp>
#include <stan/agrad/rev/op/vector_vari.hpp>
#include <stan/agrad/rev/numeric_limits.hpp>
#include <stan/agrad/rev/operator_greater_than.hpp>

#include <stan/math.hpp>
#include <stan/math/error_handling.hpp>
#include <stan/math/functions/Phi.hpp>

#include <stan/math/functions/log_sum_exp.hpp>
#include <boost/math/special_functions/digamma.hpp>

namespace stan {

  namespace agrad {
    
    using stan::math::check_not_nan;
    using stan::math::check_greater_or_equal;

    // OTHER FUNCTIONS: stan/math/special_functions.hpp implementations




    /**
     * Return the value of the specified variable.  
     *
     * <p>This function is used internally by auto-dif functions along
     * with <code>stan::math::value_of(T x)</code> to extract the
     * <code>double</code> value of either a scalar or an auto-dif
     * variable.  This function will be called when the argument is a
     * <code>stan::agrad::var</code> even if the function is not
     * referred to by namespace because of argument-dependent lookup.
     *
     * @param v Variable.
     * @return Value of variable.
     */
    inline double value_of(const agrad::var& v) {
      return v.vi_->val_;
    }

    /**
     * Return 1 if the argument is unequal to zero and 0 otherwise.
     *
     * @param x Value.
     * @return 1 if argument is equal to zero and 0 otherwise.
     */
    inline int as_bool(const agrad::var& v) {
      return 0.0 != v.vi_->val_;
    }

  } // namespace math

} // namespace stan


#endif
