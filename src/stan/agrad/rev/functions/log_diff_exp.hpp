#ifndef STAN__AGRAD__REV__FUNCTIONS__LOG_DIFF_EXP_HPP
#define STAN__AGRAD__REV__FUNCTIONS__LOG_DIFF_EXP_HPP

#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/calculate_chain.hpp>
#include <stan/agrad/rev/internal/vv_vari.hpp>
#include <stan/agrad/rev/internal/vd_vari.hpp>
#include <stan/agrad/rev/internal/dv_vari.hpp>
#include <stan/agrad/rev/internal/vector_vari.hpp>
#include <stan/agrad/rev/operators/operator_greater_than.hpp>
#include <stan/agrad/rev/operators/operator_not_equal.hpp>
#include <stan/math/functions/log_diff_exp.hpp>
#include <boost/math/special_functions/expm1.hpp>

namespace stan {
  namespace agrad {

    namespace {
      class log_diff_exp_vv_vari : public op_vv_vari {
      public:
        log_diff_exp_vv_vari(vari* avi, vari* bvi) :
          op_vv_vari(stan::math::log_diff_exp(avi->val_, bvi->val_),
                     avi, bvi) {
        }
        void chain() {
          avi_->adj_ += adj_ * calculate_chain(avi_->val_, val_);
          bvi_->adj_ -= adj_ / boost::math::expm1(avi_->val_ - bvi_->val_);
        }
      };
      class log_diff_exp_vd_vari : public op_vd_vari {
      public:
        log_diff_exp_vd_vari(vari* avi, double b) :
          op_vd_vari(stan::math::log_diff_exp(avi->val_, b),
                     avi, b) {
        }
        void chain() {
          avi_->adj_ += adj_ * calculate_chain(avi_->val_, val_);
        }
      };
      class log_diff_exp_dv_vari : public op_dv_vari {
      public:
        log_diff_exp_dv_vari(double a, vari* bvi) :
          op_dv_vari(stan::math::log_diff_exp(a, bvi->val_),
                     a, bvi) {
        }
        void chain() {
          bvi_->adj_ -= adj_ / boost::math::expm1(ad_ - bvi_->val_);
        }
      };
    }

    /**
     * Returns the log sum of exponentials.
     */
    inline var log_diff_exp(const stan::agrad::var& a,
                            const stan::agrad::var& b) {
      return var(new log_diff_exp_vv_vari(a.vi_, b.vi_));
    }
    /**
     * Returns the log sum of exponentials.
     */
    inline var log_diff_exp(const stan::agrad::var& a,
                            const double& b) {
      return var(new log_diff_exp_vd_vari(a.vi_, b));
    }
    /**
     * Returns the log sum of exponentials.
     */
    inline var log_diff_exp(const double& a,
                            const stan::agrad::var& b) {
      return var(new log_diff_exp_dv_vari(a, b.vi_));
    }

  }
}
#endif
