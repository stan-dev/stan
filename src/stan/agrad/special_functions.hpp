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


  } // namespace math

} // namespace stan


#endif
