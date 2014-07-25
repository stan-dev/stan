#ifndef STAN__AGRAD__REV__INTERNAL__DVV_VARI_HPP
#define STAN__AGRAD__REV__INTERNAL__DVV_VARI_HPP

#include <stan/agrad/rev/vari.hpp>

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
