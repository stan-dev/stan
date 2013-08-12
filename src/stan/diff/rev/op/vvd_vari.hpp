#ifndef __STAN__DIFF__REV__OP__VVD_VARI_HPP__
#define __STAN__DIFF__REV__OP__VVD_VARI_HPP__

#include <stan/diff/rev/vari.hpp>

namespace stan {
  namespace diff {

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
