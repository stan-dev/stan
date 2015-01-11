#ifndef STAN__AGRAD__REV__FUNCTIONS__LOG_SUM_EXP_HPP
#define STAN__AGRAD__REV__FUNCTIONS__LOG_SUM_EXP_HPP

#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/calculate_chain.hpp>
#include <stan/agrad/rev/internal/vv_vari.hpp>
#include <stan/agrad/rev/internal/vd_vari.hpp>
#include <stan/agrad/rev/internal/dv_vari.hpp>
#include <stan/agrad/rev/internal/vector_vari.hpp>
#include <stan/agrad/rev/operators/operator_greater_than.hpp>
#include <stan/agrad/rev/operators/operator_not_equal.hpp>
#include <stan/math/functions/log_sum_exp.hpp>

namespace stan {
  namespace agrad {

    namespace {
      double log_sum_exp_as_double(const std::vector<var>& x) {
        using std::numeric_limits;
        using std::exp;
        using std::log;
        double max = -numeric_limits<double>::infinity();
        for (size_t i = 0; i < x.size(); ++i) 
          if (x[i] > max) 
            max = x[i].val();
        double sum = 0.0;
        for (size_t i = 0; i < x.size(); ++i) 
          if (x[i] != -numeric_limits<double>::infinity()) 
            sum += exp(x[i].val() - max);
        return max + log(sum);
      }

      class log_sum_exp_vv_vari : public op_vv_vari {
      public:
        log_sum_exp_vv_vari(vari* avi, vari* bvi) :
          op_vv_vari(stan::math::log_sum_exp(avi->val_, bvi->val_),
                     avi, bvi) {
        }
        void chain() {
          avi_->adj_ += adj_ * calculate_chain(avi_->val_, val_);
          bvi_->adj_ += adj_ * calculate_chain(bvi_->val_, val_);
        }
      };
      class log_sum_exp_vd_vari : public op_vd_vari {
      public:
        log_sum_exp_vd_vari(vari* avi, double b) :
          op_vd_vari(stan::math::log_sum_exp(avi->val_, b),
                     avi, b) {
        }
        void chain() {
          avi_->adj_ += adj_ * calculate_chain(avi_->val_, val_);
        }
      };
      class log_sum_exp_dv_vari : public op_dv_vari {
      public:
        log_sum_exp_dv_vari(double a, vari* bvi) :
          op_dv_vari(stan::math::log_sum_exp(a, bvi->val_),
                     a, bvi) {
        }
        void chain() {
          bvi_->adj_ += adj_ * calculate_chain(bvi_->val_, val_);
        }
      };

      class log_sum_exp_vector_vari : public op_vector_vari {
      public:
        log_sum_exp_vector_vari(const std::vector<var>& x) :
          op_vector_vari(log_sum_exp_as_double(x), x) {
        }
        void chain() {
          for (size_t i = 0; i < size_; ++i) {
            vis_[i]->adj_ += adj_ * calculate_chain(vis_[i]->val_, val_);
          }
        }
      };
    }

    /**
     * Returns the log sum of exponentials.
     */
    inline var log_sum_exp(const stan::agrad::var& a,
                           const stan::agrad::var& b) {
      return var(new log_sum_exp_vv_vari(a.vi_, b.vi_));
    }
    /**
     * Returns the log sum of exponentials.
     */
    inline var log_sum_exp(const stan::agrad::var& a,
                           const double& b) {
      return var(new log_sum_exp_vd_vari(a.vi_, b));
    }
    /**
     * Returns the log sum of exponentials.
     */
    inline var log_sum_exp(const double& a,
                           const stan::agrad::var& b) {
      return var(new log_sum_exp_dv_vari(a, b.vi_));
    }
    /**
     * Returns the log sum of exponentials.
     */
    inline var log_sum_exp(const std::vector<var>& x) {
      return var(new log_sum_exp_vector_vari(x));
    }

  }
}
#endif
