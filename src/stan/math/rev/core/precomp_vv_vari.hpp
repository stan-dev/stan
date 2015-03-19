#ifndef STAN__MATH__REV__CORE__PRECOMP_VV_VARI_HPP
#define STAN__MATH__REV__CORE__PRECOMP_VV_VARI_HPP

#include <stan/math/rev/core/vari.hpp>
#include <stan/math/rev/core/vv_vari.hpp>

namespace stan {
  namespace agrad {

    // use for single precomputed partials
    class precomp_vv_vari : public op_vv_vari {
    protected:
      double da_;
      double db_;
    public:
      precomp_vv_vari(double val,
                       vari* avi, vari* bvi,
                       double da, double db)
        : op_vv_vari(val,avi,bvi),
          da_(da),
          db_(db) {
      }
      void chain() {
        avi_->adj_ += adj_ * da_;
        bvi_->adj_ += adj_ * db_;
      }
    };

  }
}
#endif
