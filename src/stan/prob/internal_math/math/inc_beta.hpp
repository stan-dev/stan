#ifndef __STAN__PROB__INTERNAL_MATH__MATH__INC_BETA_HPP__
#define __STAN__PROB__INTERNAL_MATH__MATH__INC_BETA_HPP__

#include <boost/math/special_functions/beta.hpp>

namespace stan {

  namespace math {

    inline double inc_beta(const double& a,
                           const double& b,
                           const double& x) {
      using boost::math::beta;
      return beta(a,b,x);
    }
  }
}
#endif
