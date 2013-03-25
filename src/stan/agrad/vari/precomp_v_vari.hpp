#ifndef __STAN__AGRAD__VARI__PRECOMP_V_VARI_HPP__
#define __STAN__AGRAD__VARI__PRECOMP_V_VARI_HPP__

#include <stan/agrad/vari/vari.hpp>
#include <stan/agrad/vari/op_v_vari.hpp>

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
