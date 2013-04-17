#ifndef __STAN__AGRAD__REV__MATRIX__SQUARED_DIST_HPP__
#define __STAN__AGRAD__REV__MATRIX__SQUARED_DIST_HPP__

#include <vector>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/math/matrix/validate_vector.hpp>
#include <stan/math/matrix/validate_matching_sizes.hpp>
#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/vari.hpp>
#include <stan/agrad/rev/sqrt.hpp>
#include <stan/agrad/rev/matrix/typedefs.hpp>

namespace stan {
  namespace agrad {
    namespace {
      class squared_dist_vv_vari : public vari {
      protected:
        vari** v1_;
        vari** v2_;
        size_t length_;
        
        template<int R1,int C1,int R2,int C2>
        inline static double var_squared_dist(const Eigen::Matrix<var,R1,C1> &v1,
                                              const Eigen::Matrix<var,R2,C2> &v2) {
          double result = 0;
          for (size_t i = 0; i < v1.size(); i++) {
            double diff = v1(i).vi_->val_ - v2(i).vi_->val_;
            result += diff*diff;
          }
          return result;
        }
      public:
        template<int R1,int C1,int R2,int C2>
        squared_dist_vv_vari(const Eigen::Matrix<var,R1,C1> &v1,
                             const Eigen::Matrix<var,R2,C2> &v2) 
        : vari(var_squared_dist(v1, v2)), length_(v1.size())
        {
          v1_ = (vari**)memalloc_.alloc(length_*sizeof(vari*));
          for (size_t i = 0; i < length_; i++)
            v1_[i] = v1(i).vi_;
          
          v2_ = (vari**)memalloc_.alloc(length_*sizeof(vari*));
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
      class squared_dist_vd_vari : public vari {
      protected:
        vari** v1_;
        double* v2_;
        size_t length_;
        
        template<int R1,int C1,int R2,int C2>
        inline static double var_squared_dist(const Eigen::Matrix<var,R1,C1> &v1,
                                              const Eigen::Matrix<double,R2,C2> &v2) {
          double result = 0;
          for (size_t i = 0; i < v1.size(); i++) {
            double diff = v1(i).vi_->val_ - v2(i);
            result += diff*diff;
          }
          return result;
        }
      public:
        template<int R1,int C1,int R2,int C2>
        squared_dist_vd_vari(const Eigen::Matrix<var,R1,C1> &v1,
                             const Eigen::Matrix<double,R2,C2> &v2) 
        : vari(var_squared_dist(v1, v2)), length_(v1.size())
        {
          v1_ = (vari**)memalloc_.alloc(length_*sizeof(vari*));
          for (size_t i = 0; i < length_; i++)
            v1_[i] = v1(i).vi_;
          
          v2_ = (double*)memalloc_.alloc(length_*sizeof(double));
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
    inline var squared_dist(const Eigen::Matrix<var, R1, C1>& v1, 
                            const Eigen::Matrix<var, R2, C2>& v2) {
      stan::math::validate_vector(v1,"squared_dist");
      stan::math::validate_vector(v2,"squared_dist");
      stan::math::validate_matching_sizes(v1,v2,"squared_dist");
      return var(new squared_dist_vv_vari(v1,v2));
    }
    template<int R1,int C1,int R2, int C2>
    inline var squared_dist(const Eigen::Matrix<var, R1, C1>& v1, 
                            const Eigen::Matrix<double, R2, C2>& v2) {
      stan::math::validate_vector(v1,"squared_dist");
      stan::math::validate_vector(v2,"squared_dist");
      stan::math::validate_matching_sizes(v1,v2,"squared_dist");
      return var(new squared_dist_vd_vari(v1,v2));
    }
    template<int R1,int C1,int R2, int C2>
    inline var squared_dist(const Eigen::Matrix<double, R1, C1>& v1, 
                            const Eigen::Matrix<var, R2, C2>& v2) {
      stan::math::validate_vector(v1,"squared_dist");
      stan::math::validate_vector(v2,"squared_dist");
      stan::math::validate_matching_sizes(v1,v2,"squared_dist");
      return var(new squared_dist_vd_vari(v2,v1));
    }
  }
}
#endif
