#ifndef __STAN__AGRAD__REV__INTERNAL__VDD_VARI_HPP__
#define __STAN__AGRAD__REV__INTERNAL__VDD_VARI_HPP__

#include <stan/agrad/rev/vari.hpp>

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
