#ifndef __STAN__AGRAD__FUNCTIONS__OP_VVD_VARI_HPP__
#define __STAN__AGRAD__FUNCTIONS__OP_VVD_VARI_HPP__

#include <stan/agrad/vari.hpp>

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
