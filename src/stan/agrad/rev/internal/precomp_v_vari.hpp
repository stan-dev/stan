#ifndef STAN__AGRAD__REV__INTERNAL__PRECOMP_V_VARI_HPP
#define STAN__AGRAD__REV__INTERNAL__PRECOMP_V_VARI_HPP

#include <stan/agrad/rev/vari.hpp>
#include <stan/agrad/rev/internal/v_vari.hpp>

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
