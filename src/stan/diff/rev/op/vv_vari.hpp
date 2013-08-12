#ifndef __STAN__DIFF__REV__OP__VV_VARI_HPP__
#define __STAN__DIFF__REV__OP__VV_VARI_HPP__

#include <stan/diff/rev/vari.hpp>

namespace stan {
  namespace diff {
    
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
