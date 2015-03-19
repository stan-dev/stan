#ifndef STAN__MATH__REV__MAT__FUN__DOT_SELF_HPP
#define STAN__MATH__REV__MAT__FUN__DOT_SELF_HPP

#include <vector>
#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <stan/math/prim/mat/err/check_vector.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/rev/mat/fun/typedefs.hpp>

namespace stan {
  namespace agrad {

    namespace {
      class dot_self_vari : public vari {
      protected:
        vari** v_;
        size_t size_;
      public:
        dot_self_vari(vari** v, size_t size)
          : vari(var_dot_self(v,size)),
            v_(v),
            size_(size) {
        }
        template<typename Derived>
        dot_self_vari(const Eigen::DenseBase<Derived> &v) :
          vari(var_dot_self(v)), size_(v.size()) {
          v_ = (vari**)ChainableStack::memalloc_.alloc(size_*sizeof(vari*));
          for (size_t i = 0; i < size_; i++)
            v_[i] = v[i].vi_;
        }
        template <int R, int C>
        dot_self_vari(const Eigen::Matrix<var,R,C>& v) :
          vari(var_dot_self(v)), size_(v.size()) {
          v_ = (vari**) ChainableStack::memalloc_.alloc(size_ * sizeof(vari*));
          for (size_t i = 0; i < size_; ++i)
            v_[i] = v(i).vi_;
        }
        inline static double square(double x) { return x * x; }
        inline static double var_dot_self(vari** v, size_t size) {
          double sum = 0.0;
          for (size_t i = 0; i < size; ++i)
            sum += square(v[i]->val_);
          return sum;
        }
        template<typename Derived>
        double var_dot_self(const Eigen::DenseBase<Derived> &v) {
          double sum = 0.0;
          for (int i = 0; i < v.size(); ++i)
            sum += square(v(i).vi_->val_);
          return sum;
        }
        template <int R, int C>
        inline static double var_dot_self(const Eigen::Matrix<var,R,C> &v) {
          double sum = 0.0;
          for (int i = 0; i < v.size(); ++i)
            sum += square(v(i).vi_->val_);
          return sum;
        }
        virtual void chain() {
          for (size_t i = 0; i < size_; ++i)
            v_[i]->adj_ += adj_ * 2.0 * v_[i]->val_;
        }
      };
    }
    /**
     * Returns the dot product of a vector with itself.
     *
     * @param[in] v Vector.
     * @return Dot product of the vector with itself.
     * @tparam R number of rows or <code>Eigen::Dynamic</code> for
     * dynamic; one of R or C must be 1
     * @tparam C number of rows or <code>Eigen::Dyanmic</code> for
     * dynamic; one of R or C must be 1
     */
    template<int R, int C>
    inline var dot_self(const Eigen::Matrix<var, R, C>& v) {
      stan::math::check_vector("dot_self", "v", v);
      return var(new dot_self_vari(v));
    }

  }
}
#endif
