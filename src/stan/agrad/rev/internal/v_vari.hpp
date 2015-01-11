#ifndef STAN__AGRAD__REV__INTERNAL__V_VARI_HPP
#define STAN__AGRAD__REV__INTERNAL__V_VARI_HPP

#include <stan/agrad/rev/vari.hpp>

namespace stan {
  namespace agrad {
    
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
