#ifndef STAN__AGRAD__REV__FUNCTIONS__DIGAMMA_HPP
#define STAN__AGRAD__REV__FUNCTIONS__DIGAMMA_HPP

#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/zeta.hpp>
#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/internal/v_vari.hpp>
#include <stan/math/functions/trigamma.hpp>

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
