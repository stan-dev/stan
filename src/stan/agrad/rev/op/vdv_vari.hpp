#ifndef __STAN__AGRAD__REV__OP__VDV_VARI_HPP__
#define __STAN__AGRAD__REV__OP__VDV_VARI_HPP__

#include <stan/agrad/rev/vari.hpp>

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
