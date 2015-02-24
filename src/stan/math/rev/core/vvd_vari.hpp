#ifndef STAN__MATH__REV__CORE__VVD_VARI_HPP
#define STAN__MATH__REV__CORE__VVD_VARI_HPP

#include <stan/math/rev/core.hpp>

namespace stan {
  namespace agrad {

    class op_vvd_vari : public vari {
    protected:
      vari* avi_;
      vari* bvi_;
      double cd_;
    public:
      op_vvd_vari(double f, vari* avi, vari* bvi, double c) :
        vari(f),
        avi_(avi),
        bvi_(bvi),
        cd_(c) {
      }
    };

  }
}
#endif
