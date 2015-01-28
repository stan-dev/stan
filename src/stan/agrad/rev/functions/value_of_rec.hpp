#ifndef STAN__AGRAD__REV__FUNCTIONS__VALUE_OF_REC_HPP
#define STAN__AGRAD__REV__FUNCTIONS__VALUE_OF_REC_HPP

#include <stan/agrad/rev/var.hpp>

namespace stan {
  namespace agrad {

    /**
     * Return the value of the specified variable.  
     *
     * @param v Variable.
     * @return Value of variable.
     */
    inline double value_of_rec(const stan::agrad::var& v) {
      return v.vi_->val_;
    }

  }
}
#endif
