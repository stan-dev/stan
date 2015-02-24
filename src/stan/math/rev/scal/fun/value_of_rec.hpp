#ifndef STAN__MATH__REV__SCAL__FUN__VALUE_OF_REC_HPP
#define STAN__MATH__REV__SCAL__FUN__VALUE_OF_REC_HPP

#include <stan/math/rev/core.hpp>

namespace stan {
  namespace agrad {

    /**
     * Return the value of the specified variable.  
     *
     * @param v Variable.
     * @return Value of variable.
     */
    inline double value_of_rec(const var& v) {
      return v.vi_->val_;
    }

  }
}
#endif
