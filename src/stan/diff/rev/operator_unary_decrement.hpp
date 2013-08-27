#ifndef __STAN__DIFF__REV__OPERATOR_UNARY_DECREMENT_HPP__
#define __STAN__DIFF__REV__OPERATOR_UNARY_DECREMENT_HPP__

#include <stan/diff/rev/var.hpp>
#include <stan/diff/rev/op/v_vari.hpp>

namespace stan {
  namespace diff {
    
    namespace {
      class decrement_vari : public op_v_vari {
      public:
        decrement_vari(vari* avi) :
          op_v_vari(avi->val_ - 1.0, avi) {
        }
        void chain() {
          avi_->adj_ += adj_;
        }
      };
    }

    /**
     * Prefix decrement operator for variables (C++).  
     *
     * Following C++, <code>(--a)</code> is defined to behave exactly as 
     *
     * <code>a = a - 1.0)</code>
     *
     * does, but is faster and uses less memory.  In particular,
     * the result is an assignable lvalue.
     *
     * @param a Variable to decrement.
     * @return Reference the result of decrementing this input variable.
     */
    inline var& operator--(var& a) {
      a.vi_ = new decrement_vari(a.vi_);
      return a;
    }

    /**
     * Postfix decrement operator for variables (C++).  
     * 
     * Following C++, the expression <code>(a--)</code> is defined to
     * behave like the sequence of operations
     *
     * <code>var temp = a;  a = a - 1.0;  return temp;</code>
     *
     * @param a Variable to decrement.
     * @return Input variable. 
     */
    inline var operator--(var& a, int /*dummy*/) {
      var temp(a);
      a.vi_ = new decrement_vari(a.vi_);
      return temp;
    }
    
  }
}
#endif
