#ifndef STAN__AGRAD__REV__OPERATORS__OPERATOR_UNARY_NEGATIVE_HPP
#define STAN__AGRAD__REV__OPERATORS__OPERATOR_UNARY_NEGATIVE_HPP

#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/internal/v_vari.hpp>
#include <boost/math/special_functions/fpclassify.hpp>

namespace stan {
  namespace agrad {
    
    namespace {
      class neg_vari : public op_v_vari {
      public: 
        neg_vari(vari* avi) :
          op_v_vari(-(avi->val_), avi) {
        }
        void chain() {
          if (unlikely(boost::math::isnan(avi_->val_)))
            avi_->adj_ = std::numeric_limits<double>::quiet_NaN();
          else
            avi_->adj_ -= adj_;
        }
      };
    }

    /**
     * Unary negation operator for variables (C++).
     *
     * \f$\frac{d}{dx} -x = -1\f$.
     *
     * @param a Argument variable.
     * @return Negation of variable.
     */
    inline var operator-(const var& a) {
      return var(new neg_vari(a.vi_));
    }

  }
}
#endif
