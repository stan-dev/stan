#ifndef STAN__MATH__REV__CORE__DV_VARI_HPP
#define STAN__MATH__REV__CORE__DV_VARI_HPP

#include <stan/math/rev/core.hpp>

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
