#ifndef STAN__MATH__REV__CORE__VDD_VARI_HPP
#define STAN__MATH__REV__CORE__VDD_VARI_HPP

#include <stan/math/rev/core.hpp>

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
