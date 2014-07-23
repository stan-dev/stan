#ifndef STAN__AGRAD__REV__INTERNAL__VV_VARI_HPP
#define STAN__AGRAD__REV__INTERNAL__VV_VARI_HPP

#include <stan/agrad/rev/vari.hpp>

namespace stan {
  namespace agrad {
    
    class op_vv_vari : public vari {
    protected:
      vari* avi_;
      vari* bvi_;
    public:
      op_vv_vari(double f, vari* avi, vari* bvi):
        vari(f),
        avi_(avi),
        bvi_(bvi) {
      }
    };

  }
}
#endif
