#ifndef __STAN__AGRAD__FUNCTIONS__OP_VDD_VARI_HPP__
#define __STAN__AGRAD__FUNCTIONS__OP_VDD_VARI_HPP__

#include <stan/agrad/vari.hpp>

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
