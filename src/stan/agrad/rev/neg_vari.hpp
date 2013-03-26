#ifndef __STAN__AGRAD__REV__NEG_VARI_HPP__
#define __STAN__AGRAD__REV__NEG_VARI_HPP__

#include <stan/agrad/rev/op/v_vari.hpp>

namespace stan {
  namespace agrad {

    class neg_vari : public op_v_vari {
    public: 
      neg_vari(vari* avi) :
      op_v_vari(-(avi->val_), avi) {
      }
      void chain() {
        avi_->adj_ -= adj_;
      }
    };

  }
}
#endif
