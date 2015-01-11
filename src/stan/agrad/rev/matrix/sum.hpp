#ifndef STAN__AGRAD__REV__MATRIX__SUM_HPP
#define STAN__AGRAD__REV__MATRIX__SUM_HPP

#include <vector>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/vari.hpp>
#include <stan/agrad/rev/matrix/typedefs.hpp>

namespace stan {
  namespace agrad {

    namespace {
      class sum_v_vari : public vari{
      protected:
        vari** v_;
        size_t length_;
        inline static double var_sum(const var *v, size_t length) {
          double result = 0;
          for (size_t i = 0; i < length; i++)
            result += v[i].vi_->val_;
          return result;
        } 
        template<typename Derived>
        inline static double var_sum(const Eigen::DenseBase<Derived> &v) {
          double result = 0;
          for (int i = 0; i < v.size(); i++)
            result += v(i).vi_->val_;
          return result;
        } 
      public:
        template<typename Derived>
        sum_v_vari(const Eigen::DenseBase<Derived> &v) :
          vari(var_sum(v)), length_(v.size()) {
          v_ = (vari**)ChainableStack::memalloc_.alloc(length_*sizeof(vari*));
          for (size_t i = 0; i < length_; i++)
            v_[i] = v(i).vi_;
        }
        template<int R1,int C1>
        sum_v_vari(const Eigen::Matrix<var,R1,C1> &v1) :
          vari(var_sum(v1)), length_(v1.size()) {
          v_ = (vari**)ChainableStack::memalloc_.alloc(length_*sizeof(vari*));
          for (size_t i = 0; i < length_; i++)
            v_[i] = v1(i).vi_;
        }
        sum_v_vari(const var *v, size_t len) :
          vari(var_sum(v,len)), length_(len) {
          v_ = (vari**)ChainableStack::memalloc_.alloc(length_*sizeof(vari*));
          for (size_t i = 0; i < length_; i++)
            v_[i] = v[i].vi_;
        }
        virtual void chain() {
          for (size_t i = 0; i < length_; i++) {
            v_[i]->adj_ += adj_;
          }
        }
      };
    }

    /**
     * Returns the sum of the coefficients of the specified
     * matrix, column vector or row vector.
     * @param m Specified matrix or vector.
     * @return Sum of coefficients of matrix.
     */
    template <int R, int C>
    inline var sum(const Eigen::Matrix<var,R,C>& m) {
      if (m.size() == 0)
        return 0.0;
      return var(new sum_v_vari(m));
    }

  }
}
#endif
