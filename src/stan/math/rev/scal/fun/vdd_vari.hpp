#ifndef STAN__MATH__REV__SCAL__FUN__VDD_VARI_HPP
#define STAN__MATH__REV__SCAL__FUN__VDD_VARI_HPP

#include <stan/math/rev/arr/meta/vari.hpp>

namespace stan {
  namespace agrad {

    class op_vdd_vari : public vari {
    protected:
      vari* avi_;
      double bd_;
      double cd_;
    public:
      op_vdd_vari(double f, vari* avi, double b, double c) :
        vari(f),
        avi_(avi),
        bd_(b), 
        cd_(c) {
      }
    };

  }
}
#endif
