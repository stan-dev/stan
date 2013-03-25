#ifndef __STAN__AGRAD__FUNCTIONS__OP_DV_VARI_HPP__
#define __STAN__AGRAD__FUNCTIONS__OP_DV_VARI_HPP__

#include <stan/agrad/vari.hpp>

namespace stan {
  namespace agrad {
    
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
