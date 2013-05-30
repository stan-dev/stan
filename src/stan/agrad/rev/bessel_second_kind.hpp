#ifndef __STAN__AGRAD__REV__BESSEL_SECOND_KIND_HPP__
#define __STAN__AGRAD__REV__BESSEL_SECOND_KIND_HPP__

#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/op/vv_vari.hpp>
#include <stan/agrad/rev/op/vd_vari.hpp>
#include <stan/agrad/rev/op/dv_vari.hpp>
#include <stan/math/constants.hpp>
#include <stan/math/functions/bessel_second_kind.hpp>

namespace stan {
  namespace agrad {

    namespace {

      class bessel_second_kind_dv_vari : public op_dv_vari {
      public:
        bessel_second_kind_dv_vari(int a, vari* bvi) :
          op_dv_vari(stan::math::bessel_second_kind(a, bvi->val_), a, bvi) {
        }
        void chain() {
          bvi_->adj_ += adj_ * (ad_ * stan::math::bessel_second_kind(ad_, bvi_->val_) / bvi_->val_ - stan::math::bessel_second_kind(ad_ + 1, bvi_->val_));
        }
      };
    }

    inline var bessel_second_kind(const int& v, 
                                  const var& a) {
      return var(new bessel_second_kind_dv_vari(v, a.vi_));
    }

  }
}
#endif
