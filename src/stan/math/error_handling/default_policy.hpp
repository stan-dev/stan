#ifndef __STAN__MATH__ERROR_HANDLING__DEFAULT_POLICY_HPP__
#define __STAN__MATH__ERROR_HANDLING__DEFAULT_POLICY_HPP__

#include <boost/math/policies/policy.hpp>

namespace stan {
  namespace math {

    /**
     * Default error-handling policy from Boost.
     */
    typedef boost::math::policies::policy<> default_policy;
    
  }
}
#endif
