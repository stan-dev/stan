#ifndef STAN__MATH__REV__SCAL__FUN__DIGAMMA_HPP
#define STAN__MATH__REV__SCAL__FUN__DIGAMMA_HPP

#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/zeta.hpp>
#include <stan/math/rev/core/var.hpp>
#include <stan/math/rev/scal/fun/v_vari.hpp>
#include <stan/math/prim/scal/fun/trigamma.hpp>

namespace stan {
  namespace agrad {

    namespace {
      class digamma_vari : public op_v_vari {
      public:
        digamma_vari(vari* avi) :
          op_v_vari(boost::math::digamma(avi->val_), avi) {
        }
        void chain() {
          avi_->adj_ += adj_ * stan::math::trigamma(avi_->val_);
        }
      };
    }

    inline var digamma(const stan::agrad::var& a) {
      return var(new digamma_vari(a.vi_));
    }

  }
}
#endif
