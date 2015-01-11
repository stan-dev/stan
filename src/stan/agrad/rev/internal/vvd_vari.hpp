#ifndef STAN__AGRAD__REV__INTERNAL__VVD_VARI_HPP
#define STAN__AGRAD__REV__INTERNAL__VVD_VARI_HPP

#include <stan/agrad/rev/vari.hpp>

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
