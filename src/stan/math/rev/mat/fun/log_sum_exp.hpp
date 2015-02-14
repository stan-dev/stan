#ifndef STAN__MATH__REV__MAT__FUN__LOG_SUM_EXP_HPP
#define STAN__MATH__REV__MAT__FUN__LOG_SUM_EXP_HPP

#include <stan/math/rev/arr/meta/var.hpp>
#include <stan/math/rev/scal/fun/calculate_chain.hpp>
#include <stan/math/rev/scal/fun/vv_vari.hpp>
#include <stan/math/rev/scal/fun/vd_vari.hpp>
#include <stan/math/rev/scal/fun/dv_vari.hpp>
#include <stan/math/rev/scal/fun/matrix_vari.hpp>
#include <stan/math/rev/scal/fun/operator_greater_than.hpp>
#include <stan/math/rev/scal/fun/operator_not_equal.hpp>
#include <stan/math/prim/scal/fun/log_sum_exp.hpp>
#include <stan/math/prim/mat/fun/Eigen.hpp>

namespace stan {
  namespace agrad {

    namespace {

      // these function and the following class just translate
      // log_sum_exp for std::vector for Eigen::Matrix
      
      template <int R, int C>
      double log_sum_exp_as_double(const Eigen::Matrix<var,R,C>& x) {
        using std::numeric_limits;
        using std::exp;
        using std::log;
        double max = -numeric_limits<double>::infinity();
        for (int i = 0; i < x.size(); ++i) 
          if (x(i) > max) 
            max = x(i).val();
        double sum = 0.0;
        for (int i = 0; i < x.size(); ++i) 
          if (x(i) != -numeric_limits<double>::infinity()) 
            sum += exp(x(i).val() - max);
        return max + log(sum);
      }

      class log_sum_exp_matrix_vari : public op_matrix_vari {
      public:
        template <int R, int C>
        log_sum_exp_matrix_vari(const Eigen::Matrix<var,R,C>& x) :
          op_matrix_vari(log_sum_exp_as_double(x), x) {
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
     *
     * @param x matrix
     */
    template <int R, int C>
    inline var log_sum_exp(const Eigen::Matrix<var,R,C>& x) {
      return var(new log_sum_exp_matrix_vari(x));
    }

  }
}
#endif
