#ifndef __STAN__AGRAD__REV__EXP_HPP__
#define __STAN__AGRAD__REV__EXP_HPP__

#include <cmath>
#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/op/v_vari.hpp>

namespace stan {
  namespace agrad {
    
    namespace {
      class exp_vari : public op_v_vari {
      public:
        exp_vari(vari* avi) :
          op_v_vari(std::exp(avi->val_),avi) {
        }
        void chain() {
          avi_->adj_ += adj_ * val_;
        }
      };
    }

    /**
     * Return the exponentiation of the specified variable (cmath).
     *
     * @param a Variable to exponentiate.
     * @return Exponentiated variable.
     */
    inline var exp(const var& a) {
      return var(new exp_vari(a.vi_));
    }

  }
}
#endif
