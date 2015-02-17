#ifndef STAN__MATH__REV__SCAL__FUN__VV_VARI_HPP
#define STAN__MATH__REV__SCAL__FUN__VV_VARI_HPP

#include <stan/math/rev/core/vari.hpp>

namespace stan {
  namespace agrad {
    
    class op_vv_vari : public vari {
    protected:
      vari* avi_;
      vari* bvi_;
    public:
      op_vv_vari(double f, vari* avi, vari* bvi):
        vari(f),
        avi_(avi),
        bvi_(bvi) {
      }
    };

  }
}
#endif
