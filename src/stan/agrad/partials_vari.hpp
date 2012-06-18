#ifndef __STAN__AGRAD__PARTIALS_VARI_HPP__
#define __STAN__AGRAD__PARTIALS_VARI_HPP__

#include <stan/agrad/agrad.hpp>

namespace stan {

  namespace agrad {

    /**
     */
    struct partials1_vari : public vari {
      vari* subexpr1_;
      const double partial1_;
      partials1_vari(double value, vari* subexpr1, double partial1) 
        : vari(value), subexpr1_(subexpr1), partial1_(partial1) 
      { }
      void chain() {
        subexpr1_->adj_ += adj_ * partial1_;
      }
    };

    struct partials2_vari : public vari {
      vari* subexpr1_;      const double partial1_;
      vari* subexpr2_;      const double partial2_;
      partials2_vari(double value, 
                     vari* subexpr1, double partial1,
                     vari* subexpr2, double partial2) 
        : vari(value), 
          subexpr1_(subexpr1), partial1_(partial1),
          subexpr2_(subexpr2), partial2_(partial2)
      { }
      void chain() {
        subexpr1_->adj_ += adj_ * partial1_;
        subexpr2_->adj_ += adj_ * partial2_;
      }
    };

      struct partials3_vari : public vari {
      vari* subexpr1_;  const double partial1_;
      vari* subexpr2_;  const double partial2_;
      vari* subexpr3_;  const double partial3_;
      partials3_vari(double value, 
                     vari* subexpr1, double partial1,
                     vari* subexpr2, double partial2,
                     vari* subexpr3, double partial3)
        : vari(value), 
          subexpr1_(subexpr1), partial1_(partial1),
          subexpr2_(subexpr2), partial2_(partial2),
          subexpr3_(subexpr3), partial3_(partial3)
      { }
      void chain() {
        subexpr1_->adj_ += adj_ * partial1_;
        subexpr2_->adj_ += adj_ * partial2_;
        subexpr3_->adj_ += adj_ * partial3_;
      }
    };


  } 
} 


#endif
