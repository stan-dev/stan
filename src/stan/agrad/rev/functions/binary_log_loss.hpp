#ifndef STAN__AGRAD__REV__FUNCTIONS__BINARY_LOG_LOSS_HPP
#define STAN__AGRAD__REV__FUNCTIONS__BINARY_LOG_LOSS_HPP

#include <valarray>
#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/internal/v_vari.hpp>
#include <stan/math/constants.hpp>
#include <stan/math/functions/log1p.hpp>

namespace stan {
  namespace agrad {

    namespace {
      class binary_log_loss_1_vari : public op_v_vari {
      public:
        binary_log_loss_1_vari(vari* avi) :
          op_v_vari(-std::log(avi->val_),avi) {
        }
        void chain() {
          avi_->adj_ -= adj_ / avi_->val_;
        }
      };

      class binary_log_loss_0_vari : public op_v_vari {
      public:
        binary_log_loss_0_vari(vari* avi) :
          op_v_vari(-stan::math::log1p(-avi->val_),avi) {
        }
        void chain() {
          avi_->adj_ += adj_ / (1.0 - avi_->val_);
        }
      };
    }

    /**
     * The log loss function for variables (stan).
     *
     * See stan::math::binary_log_loss() for the double-based version.
     *
     * The derivative with respect to the variable \f$\hat{y}\f$ is
     *
     * \f$\frac{d}{d\hat{y}} \mbox{logloss}(1,\hat{y}) = - \frac{1}{\hat{y}}\f$, and
     *
     * \f$\frac{d}{d\hat{y}} \mbox{logloss}(0,\hat{y}) = \frac{1}{1 - \hat{y}}\f$.
     *
     *
       \f[
       \mbox{binary\_log\_loss}(y,\hat{y}) = 
       \begin{cases}
         y \log \hat{y} + (1 - y) \log (1 - \hat{y}) & \mbox{if } 0\leq \hat{y}\leq 1, 
         y\in\{ 0,1 \}\\[6pt]
         \textrm{NaN} & \mbox{if } \hat{y} = \textrm{NaN}
       \end{cases}
       \f]
   
       \f[
       \frac{\partial\,\mbox{binary\_log\_loss}(y,\hat{y})}{\partial \hat{y}} = 
       \begin{cases}
         \frac{y}{\hat{y}}-\frac{1-y}{1-\hat{y}} & \mbox{if } 0\leq \hat{y}\leq 1, 
         y\in\{ 0,1 \}\\[6pt]
         \textrm{NaN} & \mbox{if } \hat{y} = \textrm{NaN}
       \end{cases}
       \f]
     *
     * @param y Reference value.
     * @param y_hat Response variable.
     * @return Log loss of response versus reference value.
     */
    inline var binary_log_loss(const int y, const stan::agrad::var& y_hat) {
      if (y == 0)
        return var(new binary_log_loss_0_vari(y_hat.vi_));
      else
        return var(new binary_log_loss_1_vari(y_hat.vi_));
    }

  }
}
#endif
