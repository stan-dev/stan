#ifndef STAN__MATH__REV__SCAL__FUN__VD_VARI_HPP
#define STAN__MATH__REV__SCAL__FUN__VD_VARI_HPP

#include <stan/math/rev/arr/meta/vari.hpp>

namespace stan {
  namespace agrad {
    
    class op_vd_vari : public vari {
    protected:
      vari* avi_;
      double bd_;
    public:
      op_vd_vari(double f, vari* avi, double b) :
        vari(f),
        avi_(avi),
        bd_(b) {
      }
    };

  }
}
#endif
