#ifndef STAN__AGRAD__REV__FUNCTIONS__BESSEL_FIRST_KIND_HPP
#define STAN__AGRAD__REV__FUNCTIONS__BESSEL_FIRST_KIND_HPP

#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/internal/vv_vari.hpp>
#include <stan/agrad/rev/internal/vd_vari.hpp>
#include <stan/agrad/rev/internal/dv_vari.hpp>
#include <stan/math/constants.hpp>
#include <stan/math/functions/bessel_first_kind.hpp>

namespace stan {
  namespace agrad {

    namespace {

      class bessel_first_kind_dv_vari : public op_dv_vari {
      public:
        bessel_first_kind_dv_vari(int a, vari* bvi) :
          op_dv_vari(stan::math::bessel_first_kind(a, bvi->val_), a, bvi) {
        }
        void chain() {
          bvi_->adj_ += adj_ * (ad_ * stan::math::bessel_first_kind(ad_, bvi_->val_) / bvi_->val_ - stan::math::bessel_first_kind(ad_ + 1, bvi_->val_));
        }
      };
    }

    inline var bessel_first_kind(const int& v, 
                                 const var& a) {
      return var(new bessel_first_kind_dv_vari(v, a.vi_));
    }

  }
}
#endif
