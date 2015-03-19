#ifndef STAN__MATH__REV__CORE__V_VARI_HPP
#define STAN__MATH__REV__CORE__V_VARI_HPP

#include <stan/math/rev/core/vari.hpp>

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
