#ifndef __STAN__AGRAD__FUNCTIONS__OP_DVV_VARI_HPP__
#define __STAN__AGRAD__FUNCTIONS__OP_DVV_VARI_HPP__

#include <stan/agrad/vari.hpp>

namespace stan {
  namespace agrad {

    class op_dvv_vari : public vari {
    protected:
      double ad_;
      vari* bvi_;
      vari* cvi_;
    public:
      op_dvv_vari(double f, double a, vari* bvi, vari* cvi) :
        vari(f),
        ad_(a),
        bvi_(bvi),
        cvi_(cvi) {
      }
    };

  }
}
#endif
