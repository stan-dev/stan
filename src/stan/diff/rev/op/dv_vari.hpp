#ifndef __STAN__DIFF__REV__OP__DV_VARI_HPP__
#define __STAN__DIFF__REV__OP__DV_VARI_HPP__

#include <stan/diff/rev/vari.hpp>

namespace stan {
  namespace diff {
    
    class op_dv_vari : public vari {
    protected:
      double ad_;
      vari* bvi_;
    public:
      op_dv_vari(double f, double a, vari* bvi) :
        vari(f),
        ad_(a),
        bvi_(bvi) {
      }
    };

  }
}
#endif
