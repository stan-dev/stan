#ifndef STAN__AGRAD__REV__INTERNAL__PRECOMP_VVV_VARI_HPP
#define STAN__AGRAD__REV__INTERNAL__PRECOMP_VVV_VARI_HPP

#include <iostream>
#include <stan/agrad/rev/vari.hpp>
#include <stan/agrad/rev/internal/vvv_vari.hpp>

namespace stan {
  namespace agrad {

    // use for single precomputed partials
    class precomp_vvv_vari : public op_vvv_vari {
    protected:
      double da_;
      double db_;
      double dc_;
    public:
      precomp_vvv_vari(double val, 
                       vari* avi, vari* bvi, vari* cvi,
                       double da, double db, double dc)
        : op_vvv_vari(val,avi,bvi,cvi),
          da_(da),
          db_(db),
          dc_(dc) { 
      }
      void chain() {
        avi_->adj_ += adj_ * da_;
        bvi_->adj_ += adj_ * db_;
        cvi_->adj_ += adj_ * dc_;
      }
    };

  }
}
#endif
