#ifndef STAN__MATH__REV__CORE__DDV_VARI_HPP
#define STAN__MATH__REV__CORE__DDV_VARI_HPP

#include <stan/math/rev/core/vari.hpp>

namespace stan {
  namespace agrad {

    class op_ddv_vari : public vari {
    protected:
      double ad_;
      double bd_;
      vari* cvi_;
    public:
      op_ddv_vari(double f, double a, double b, vari* cvi) :
        vari(f),
        ad_(a),
        bd_(b),
        cvi_(cvi) {
      }
    };

  }
}
#endif
