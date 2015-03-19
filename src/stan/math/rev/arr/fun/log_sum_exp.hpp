#ifndef STAN__MATH__REV__ARR__FUN__LOG_SUM_EXP_HPP
#define STAN__MATH__REV__ARR__FUN__LOG_SUM_EXP_HPP

#include <stan/math/rev/core.hpp>
#include <stan/math/rev/scal/fun/calculate_chain.hpp>
#include <stan/math/prim/arr/fun/log_sum_exp.hpp>
#include <vector>

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
    inline var log_sum_exp(const std::vector<var>& x) {
      return var(new log_sum_exp_vector_vari(x));
    }

  }
}
#endif
