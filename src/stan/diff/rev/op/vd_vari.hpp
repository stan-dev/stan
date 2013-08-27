#ifndef __STAN__DIFF__REV__OP__VD_VARI_HPP__
#define __STAN__DIFF__REV__OP__VD_VARI_HPP__

#include <stan/diff/rev/vari.hpp>

namespace stan {
  namespace diff {
    
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
