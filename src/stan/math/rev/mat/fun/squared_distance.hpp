#ifndef STAN__MATH__REV__MAT__FUN__SQUARED_DISTANCE_HPP
#define STAN__MATH__REV__MAT__FUN__SQUARED_DISTANCE_HPP

#include <vector>

#include <stan/math/rev/core.hpp>
#include <stan/math/rev/scal/fun/sqrt.hpp>
#include <stan/math/rev/mat/fun/typedefs.hpp>

#include <stan/math/prim/mat/err/check_vector.hpp>
#include <stan/math/prim/mat/err/check_matching_sizes.hpp>
#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <stan/math/prim/mat/meta/index_type.hpp>
#include <stan/math/prim/arr/meta/index_type.hpp>
#include <stan/math/prim/scal/meta/index_type.hpp>


namespace stan {

  namespace agrad {

    namespace {

      class squared_distance_vv_vari : public vari {
      protected:
        vari** v1_;
        vari** v2_;
        size_t length_;

        template <int R1,int C1,int R2,int C2>
        inline static double
        var_squared_distance(const Eigen::Matrix<var,R1,C1> &v1,
                             const Eigen::Matrix<var,R2,C2> &v2) {
          using Eigen::Matrix;
          using stan::math::index_type;
          typedef typename index_type<Matrix<var,R1,R2> >::type idx_t;
          double result = 0;
          for (idx_t i = 0; i < v1.size(); i++) {
            double diff = v1(i).vi_->val_ - v2(i).vi_->val_;
            result += diff*diff;
          }
          return result;
        }
      public:
        template<int R1,int C1,int R2,int C2>
        squared_distance_vv_vari(const Eigen::Matrix<var,R1,C1> &v1,
                             const Eigen::Matrix<var,R2,C2> &v2)
        : vari(var_squared_distance(v1, v2)), length_(v1.size())
        {
          v1_ = (vari**)ChainableStack::memalloc_.alloc(length_*sizeof(vari*));
          for (size_t i = 0; i < length_; i++)
            v1_[i] = v1(i).vi_;

          v2_ = (vari**)ChainableStack::memalloc_.alloc(length_*sizeof(vari*));
          for (size_t i = 0; i < length_; i++)
            v2_[i] = v2(i).vi_;
        }
        virtual void chain() {
          for (size_t i = 0; i < length_; i++) {
            double di = 2 * adj_ * (v1_[i]->val_ - v2_[i]->val_);
            v1_[i]->adj_ += di;
            v2_[i]->adj_ -= di;
          }
        }
      };
      class squared_distance_vd_vari : public vari {
      protected:
        vari** v1_;
        double* v2_;
        size_t length_;

        template<int R1,int C1,int R2,int C2>
        inline static double
        var_squared_distance(const Eigen::Matrix<var,R1,C1> &v1,
                             const Eigen::Matrix<double,R2,C2> &v2) {

          using Eigen::Matrix;
          using stan::math::index_type;
          typedef typename index_type<Matrix<double,R1,C1> >::type idx_t;

          double result = 0;
          for (idx_t i = 0; i < v1.size(); i++) {
            double diff = v1(i).vi_->val_ - v2(i);
            result += diff*diff;
          }
          return result;
        }
      public:
        template<int R1,int C1,int R2,int C2>
        squared_distance_vd_vari(const Eigen::Matrix<var,R1,C1> &v1,
                                 const Eigen::Matrix<double,R2,C2> &v2)
        : vari(var_squared_distance(v1, v2)), length_(v1.size())
        {
          v1_ = (vari**)ChainableStack::memalloc_.alloc(length_*sizeof(vari*));
          for (size_t i = 0; i < length_; i++)
            v1_[i] = v1(i).vi_;

          v2_ = (double*)ChainableStack::memalloc_.alloc(length_*sizeof(double));
          for (size_t i = 0; i < length_; i++)
            v2_[i] = v2(i);
        }
        virtual void chain() {
          for (size_t i = 0; i < length_; i++) {
            v1_[i]->adj_ += 2 * adj_ * (v1_[i]->val_ - v2_[i]);
          }
        }
      };
    }

    template<int R1,int C1,int R2, int C2>
    inline var squared_distance(const Eigen::Matrix<var, R1, C1>& v1,
                                const Eigen::Matrix<var, R2, C2>& v2) {
      stan::math::check_vector("squared_distance", "v1", v1);
      stan::math::check_vector("squared_distance", "v2", v2);
      stan::math::check_matching_sizes("squared_distance",
                                                 "v1", v1,
                                                 "v2", v2);
      return var(new squared_distance_vv_vari(v1,v2));
    }
    template<int R1,int C1,int R2, int C2>
    inline var squared_distance(const Eigen::Matrix<var, R1, C1>& v1,
                                const Eigen::Matrix<double, R2, C2>& v2) {
      stan::math::check_vector("squared_distance", "v1", v1);
      stan::math::check_vector("squared_distance", "v2", v2);
      stan::math::check_matching_sizes("squared_distance",
                                                 "v1", v1,
                                                 "v2", v2);
      return var(new squared_distance_vd_vari(v1,v2));
    }
    template<int R1,int C1,int R2, int C2>
    inline var squared_distance(const Eigen::Matrix<double, R1, C1>& v1,
                                const Eigen::Matrix<var, R2, C2>& v2) {
      stan::math::check_vector("squared_distance", "v1", v1);
      stan::math::check_vector("squared_distance", "v2", v2);
      stan::math::check_matching_sizes("squared_distance",
                                                 "v1", v1,
                                                 "v2", v2);
      return var(new squared_distance_vd_vari(v2,v1));
    }
  }
}
#endif
