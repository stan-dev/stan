#ifndef __STAN__DIFF__REV__PRECOMP_V_VARI_HPP__
#define __STAN__DIFF__REV__PRECOMP_V_VARI_HPP__

#include <stan/diff/rev/vari.hpp>
#include <stan/diff/rev/op/v_vari.hpp>

namespace stan {
  namespace diff {

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
