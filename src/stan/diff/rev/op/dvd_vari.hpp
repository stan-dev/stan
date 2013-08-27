#ifndef __STAN__DIFF__REV__OP__DVD_VARI_HPP__
#define __STAN__DIFF__REV__OP__DVD_VARI_HPP__

#include <stan/diff/rev/vari.hpp>

namespace stan {
  namespace diff {

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
