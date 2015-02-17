#ifndef STAN__MATH__REV__SCAL__FUN__DVD_VARI_HPP
#define STAN__MATH__REV__SCAL__FUN__DVD_VARI_HPP

#include <stan/math/rev/arr/meta/vari.hpp>

namespace stan {
  namespace agrad {

    class op_dvd_vari : public vari {
    protected:
      double ad_;
      vari* bvi_;
      double cd_;
    public:
      op_dvd_vari(double f, double a, vari* bvi, double c) :
        vari(f),
        ad_(a),
        bvi_(bvi),
        cd_(c) {
      }
    };

  }
}
#endif
