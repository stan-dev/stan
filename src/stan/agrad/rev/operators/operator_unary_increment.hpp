#ifndef STAN__AGRAD__REV__OPERATORS__OPERATOR_UNARY_INCREMENT_HPP
#define STAN__AGRAD__REV__OPERATORS__OPERATOR_UNARY_INCREMENT_HPP

#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/internal/v_vari.hpp>

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

    /**
     * Postfix increment operator for variables (C++).  
     *
     * Following C++, the expression <code>(a++)</code> is defined to behave like
     * the sequence of operations
     *
     * <code>var temp = a;  a = a + 1.0;  return temp;</code>
     *
     * @param a Variable to increment.
     * @return Input variable. 
     */
    inline var operator++(var& a, int /*dummy*/) {
      var temp(a);
      a.vi_ = new increment_vari(a.vi_);
      return temp;
    }

  }
}
#endif
