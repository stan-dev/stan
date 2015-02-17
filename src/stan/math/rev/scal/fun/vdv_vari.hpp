#ifndef STAN__MATH__REV__SCAL__FUN__VDV_VARI_HPP
#define STAN__MATH__REV__SCAL__FUN__VDV_VARI_HPP

#include <stan/math/rev/arr/meta/vari.hpp>

namespace stan {
  namespace agrad {

    class op_vdv_vari : public vari {
    protected:
      vari* avi_;
      double bd_;
      vari* cvi_;
    public:
      op_vdv_vari(double f, vari* avi, double b, vari* cvi) :
        vari(f),
        avi_(avi),
        bd_(b), 
        cvi_(cvi) {
      }
    };

  }
}
#endif
