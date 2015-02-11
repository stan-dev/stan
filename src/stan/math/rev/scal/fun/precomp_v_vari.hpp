#ifndef STAN__AGRAD__REV__INTERNAL__PRECOMP_V_VARI_HPP
#define STAN__AGRAD__REV__INTERNAL__PRECOMP_V_VARI_HPP

#include <stan/math/rev/arr/meta/vari.hpp>
#include <stan/math/rev/scal/fun/v_vari.hpp>

namespace stan {
  namespace agrad {

    // use for single precomputed partials
    class precomp_v_vari : public op_v_vari {
    protected:
      double da_;
    public:
      precomp_v_vari(double val, vari* avi, double da)
        : op_v_vari(val,avi),
          da_(da) { 
      }
      void chain() {
        avi_->adj_ += adj_ * da_;
      }
    };

  }
}
#endif
