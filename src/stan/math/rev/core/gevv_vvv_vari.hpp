#ifndef STAN__MATH__REV__CORE__GEVV_VVV_VARI_HPP
#define STAN__MATH__REV__CORE__GEVV_VVV_VARI_HPP

#include <stan/math/rev/core/vari.hpp>
#include <stan/math/rev/core/var.hpp>
#include <stan/math/rev/core/chainablestack.hpp>

namespace stan {
  namespace agrad {

    class gevv_vvv_vari : public stan::agrad::vari {
    protected:
      stan::agrad::vari* alpha_;
      stan::agrad::vari** v1_;
      stan::agrad::vari** v2_;
      double dotval_;
      size_t length_;
      inline static double eval_gevv(const stan::agrad::var* alpha,
                                     const stan::agrad::var* v1, int stride1,
                                     const stan::agrad::var* v2, int stride2,
                                     size_t length, double *dotprod) {
        double result = 0;
        for (size_t i = 0; i < length; i++)
          result += v1[i*stride1].vi_->val_ * v2[i*stride2].vi_->val_;
        *dotprod = result;
        return alpha->vi_->val_ * result;
      }

    public:
      gevv_vvv_vari(const stan::agrad::var* alpha,
                    const stan::agrad::var* v1, int stride1,
                    const stan::agrad::var* v2, int stride2, size_t length) :
        vari(eval_gevv(alpha, v1, stride1, v2, stride2, length, &dotval_)),
        length_(length) {
        alpha_ = alpha->vi_;
        v1_ = reinterpret_cast<stan::agrad::vari**>
          (stan::agrad::ChainableStack::memalloc_
           .alloc(2 * length_ * sizeof(stan::agrad::vari*)));
        v2_ = v1_ + length_;
        for (size_t i = 0; i < length_; i++)
          v1_[i] = v1[i*stride1].vi_;
        for (size_t i = 0; i < length_; i++)
          v2_[i] = v2[i*stride2].vi_;
      }
      virtual ~gevv_vvv_vari() {}
      void chain() {
        const double adj_alpha = adj_ * alpha_->val_;
        for (size_t i = 0; i < length_; i++) {
          v1_[i]->adj_ += adj_alpha * v2_[i]->val_;
          v2_[i]->adj_ += adj_alpha * v1_[i]->val_;
        }
        alpha_->adj_ += adj_ * dotval_;
      }
    };

  }
}

#endif
