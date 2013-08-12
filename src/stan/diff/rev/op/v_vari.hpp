#ifndef __STAN__DIFF__REV__OP__V_VARI_HPP__
#define __STAN__DIFF__REV__OP__V_VARI_HPP__

#include <stan/diff/rev/vari.hpp>

namespace stan {
  namespace diff {
    
    class op_v_vari : public vari {
    protected:
      vari* avi_;
    public:
      op_v_vari(double f, vari* avi) :
        vari(f),
        avi_(avi) {
      }
    };

  }
}
#endif
