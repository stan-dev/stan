#ifndef __STAN__AGRAD__REV__INTERNAL__VD_VARI_HPP__
#define __STAN__AGRAD__REV__INTERNAL__VD_VARI_HPP__

#include <stan/agrad/rev/vari.hpp>

namespace stan {
  namespace agrad {
    
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
