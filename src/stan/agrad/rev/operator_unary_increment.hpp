#ifndef __STAN__AGRAD__REV__OPERATOR_UNARY_INCREMENT_HPP__
#define __STAN__AGRAD__REV__OPERATOR_UNARY_INCREMENT_HPP__

#include <stan/agrad/rev/op/v_vari.hpp>

namespace stan {
  namespace agrad {
    
    namespace {
      class increment_vari : public op_v_vari {
      public:
        increment_vari(vari* avi) :
          op_v_vari(avi->val_ + 1.0, avi) {
        }
        void chain() {
          avi_->adj_ += adj_;
        }
      };
    }

    /**
     * Prefix increment operator for variables (C++).  Following C++,
     * (++a) is defined to behave exactly as (a = a + 1.0) does,
     * but is faster and uses less memory.  In particular, the
     * result is an assignable lvalue.
     *
     * @param a Variable to increment.
     * @return Reference the result of incrementing this input variable.
     */
    inline var& operator++(var& a) {
      a.vi_ = new increment_vari(a.vi_);
      return a;
    }

  }
}
#endif
