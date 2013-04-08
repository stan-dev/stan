#ifndef __STAN__AGRAD__REV__MATRIX__DOT_PRODUCT_HPP__
#define __STAN__AGRAD__REV__MATRIX__DOT_PRODUCT_HPP__

#include <vector>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/math/matrix/validate_vector.hpp>
#include <stan/math/matrix/validate_matching_sizes.hpp>
#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/vari.hpp>
#include <stan/agrad/rev/matrix/typedefs.hpp>

namespace stan {
  namespace agrad {

    namespace {
      class dot_product_vv_vari : public vari {
      protected:
        vari** v1_;
        vari** v2_;
        size_t length_;
        inline static double var_dot(const var* v1, const var* v2,
                                     size_t length) {
          double result = 0;
          for (size_t i = 0; i < length; i++)
            result += v1[i].vi_->val_ * v2[i].vi_->val_;
          return result;
        }
        template<typename Derived1,typename Derived2>
        inline static double var_dot(const Eigen::DenseBase<Derived1> &v1,
                                     const Eigen::DenseBase<Derived2> &v2) {
          double result = 0;
          for (int i = 0; i < v1.size(); i++)
            result += v1[i].vi_->val_ * v2[i].vi_->val_;
          return result;
        }
        inline static double var_dot(vari** v1, vari** v2, size_t length) {
          double result = 0;
          for (size_t i = 0; i < length; ++i)
            result += v1[i]->val_ * v2[i]->val_;
          return result;
        }
      public:
        dot_product_vv_vari(vari** v1, vari** v2, size_t length)
          : vari(var_dot(v1,v2,length)),
            v1_(v1), 
            v2_(v2), 
            length_(length) {

        }
        dot_product_vv_vari(const var* v1, const var* v2, size_t length,
                            dot_product_vv_vari* shared_v1 = NULL,
                            dot_product_vv_vari* shared_v2 = NULL) : 
          vari(var_dot(v1, v2, length)), length_(length) {
          if (shared_v1 == NULL) {
            v1_ = (vari**)memalloc_.alloc(length_*sizeof(vari*));
            for (size_t i = 0; i < length_; i++)
              v1_[i] = v1[i].vi_;
          }
          else {
            v1_ = shared_v1->v1_;
          }
          if (shared_v2 == NULL) {
            v2_ = (vari**)memalloc_.alloc(length_*sizeof(vari*));
            for (size_t i = 0; i < length_; i++)
              v2_[i] = v2[i].vi_;
          }
          else {
            v2_ = shared_v2->v2_;
          }
        }
        template<typename Derived1,typename Derived2>
        dot_product_vv_vari(const Eigen::DenseBase<Derived1> &v1,
                            const Eigen::DenseBase<Derived2> &v2,
                            dot_product_vv_vari* shared_v1 = NULL,
                            dot_product_vv_vari* shared_v2 = NULL) : 
          vari(var_dot(v1, v2)), length_(v1.size()) {
          if (shared_v1 == NULL) {
            v1_ = (vari**)memalloc_.alloc(length_*sizeof(vari*));
            for (size_t i = 0; i < length_; i++)
              v1_[i] = v1[i].vi_;
          }
          else {
            v1_ = shared_v1->v1_;
          }
          if (shared_v2 == NULL) {
            v2_ = (vari**)memalloc_.alloc(length_*sizeof(vari*));
            for (size_t i = 0; i < length_; i++)
              v2_[i] = v2[i].vi_;
          }
          else {
            v2_ = shared_v2->v2_;
          }
        }
        template<int R1,int C1,int R2,int C2>
        dot_product_vv_vari(const Eigen::Matrix<var,R1,C1> &v1,
                            const Eigen::Matrix<var,R2,C2> &v2,
                            dot_product_vv_vari* shared_v1 = NULL,
                            dot_product_vv_vari* shared_v2 = NULL) : 
          vari(var_dot(v1, v2)), length_(v1.size()) {
          if (shared_v1 == NULL) {
            v1_ = (vari**)memalloc_.alloc(length_*sizeof(vari*));
            for (size_t i = 0; i < length_; i++)
              v1_[i] = v1[i].vi_;
          }
          else {
            v1_ = shared_v1->v1_;
          }
          if (shared_v2 == NULL) {
            v2_ = (vari**)memalloc_.alloc(length_*sizeof(vari*));
            for (size_t i = 0; i < length_; i++)
              v2_[i] = v2[i].vi_;
          }
          else {
            v2_ = shared_v2->v2_;
          }
        }
        virtual void chain() {
          for (size_t i = 0; i < length_; i++) {
            v1_[i]->adj_ += adj_ * v2_[i]->val_;
            v2_[i]->adj_ += adj_ * v1_[i]->val_;
          }
        }
      };

      class dot_product_vd_vari : public vari {
      protected:
        vari** v1_;
        double* v2_;
        size_t length_;
        inline static double var_dot(const var* v1, const double* v2,
                                     size_t length) {
          double result = 0;
          for (size_t i = 0; i < length; i++)
            result += v1[i].vi_->val_ * v2[i];
          return result;
        }
        template<typename Derived1,typename Derived2>
        inline static double var_dot(const Eigen::DenseBase<Derived1> &v1,
                                     const Eigen::DenseBase<Derived2> &v2) {
          double result = 0;
          for (int i = 0; i < v1.size(); i++)
            result += v1[i].vi_->val_ * v2[i];
          return result;
        }
      public:
        dot_product_vd_vari(const var* v1, const double* v2, size_t length,
                            dot_product_vd_vari *shared_v1 = NULL,
                            dot_product_vd_vari *shared_v2 = NULL) : 
          vari(var_dot(v1, v2, length)), length_(length) {
          if (shared_v1 == NULL) {
            v1_ = (vari**)memalloc_.alloc(length_*sizeof(vari*));
            for (size_t i = 0; i < length_; i++)
              v1_[i] = v1[i].vi_;
          } else {
            v1_ = shared_v1->v1_;
          }
          if (shared_v2 == NULL) {
            v2_ = (double*)memalloc_.alloc(length_*sizeof(double));
            for (size_t i = 0; i < length_; i++)
              v2_[i] = v2[i];
          } else {
            v2_ = shared_v2->v2_;
          }
        }
        template<typename Derived1,typename Derived2>
        dot_product_vd_vari(const Eigen::DenseBase<Derived1> &v1,
                            const Eigen::DenseBase<Derived2> &v2,
                            dot_product_vd_vari *shared_v1 = NULL,
                            dot_product_vd_vari *shared_v2 = NULL) : 
          vari(var_dot(v1, v2)), length_(v1.size()) {
          if (shared_v1 == NULL) {
            v1_ = (vari**)memalloc_.alloc(length_*sizeof(vari*));
            for (size_t i = 0; i < length_; i++)
              v1_[i] = v1[i].vi_;
          } else {
            v1_ = shared_v1->v1_;
          }
          if (shared_v2 == NULL) {
            v2_ = (double*)memalloc_.alloc(length_*sizeof(double));
            for (size_t i = 0; i < length_; i++)
              v2_[i] = v2[i];
          } else {
            v2_ = shared_v2->v2_;
          }
        }
        template<int R1,int C1,int R2,int C2>
        dot_product_vd_vari(const Eigen::Matrix<var,R1,C1> &v1,
                            const Eigen::Matrix<double,R2,C2> &v2,
                            dot_product_vd_vari *shared_v1 = NULL,
                            dot_product_vd_vari *shared_v2 = NULL) : 
          vari(var_dot(v1, v2)), length_(v1.size()) {
          if (shared_v1 == NULL) {
            v1_ = (vari**)memalloc_.alloc(length_*sizeof(vari*));
            for (size_t i = 0; i < length_; i++)
              v1_[i] = v1[i].vi_;
          } else {
            v1_ = shared_v1->v1_;
          }
          if (shared_v2 == NULL) {
            v2_ = (double*)memalloc_.alloc(length_*sizeof(double));
            for (size_t i = 0; i < length_; i++)
              v2_[i] = v2[i];
          } else {
            v2_ = shared_v2->v2_;
          }
        }
        virtual void chain() {
          for (size_t i = 0; i < length_; i++) {
            v1_[i]->adj_ += adj_ * v2_[i];
          }
        }
      };
    }

    /**
     * Returns the dot product.
     *
     * @param[in] v1 First column vector.
     * @param[in] v2 Second column vector.
     * @return Dot product of the vectors.
     * @throw std::domain_error if length of v1 is not equal to length of v2.
     */
    template<int R1,int C1,int R2, int C2>
    inline var dot_product(const Eigen::Matrix<var, R1, C1>& v1, 
                           const Eigen::Matrix<var, R2, C2>& v2) {
      stan::math::validate_vector(v1,"dot_product");
      stan::math::validate_vector(v2,"dot_product");
      stan::math::validate_matching_sizes(v1,v2,"dot_product");
      return var(new dot_product_vv_vari(v1,v2));
    }
    /**
     * Returns the dot product.
     *
     * @param[in] v1 First column vector.
     * @param[in] v2 Second column vector.
     * @return Dot product of the vectors.
     * @throw std::domain_error if length of v1 is not equal to length of v2
     * or either v1 or v2 are not vectors.
     */
    template<int R1,int C1,int R2, int C2>
    inline var dot_product(const Eigen::Matrix<var, R1, C1>& v1, 
                           const Eigen::Matrix<double, R2, C2>& v2) {
      stan::math::validate_vector(v1,"dot_product");
      stan::math::validate_vector(v2,"dot_product");
      stan::math::validate_matching_sizes(v1,v2,"dot_product");
      return var(new dot_product_vd_vari(v1,v2));
    }
    /**
     * Returns the dot product.
     *
     * @param[in] v1 First column vector.
     * @param[in] v2 Second column vector.
     * @return Dot product of the vectors.
     * @throw std::domain_error if length of v1 is not equal to length of v2
     * or either v1 or v2 are not vectors.
     */
    template<int R1,int C1,int R2, int C2>
    inline var dot_product(const Eigen::Matrix<double, R1, C1>& v1, 
                           const Eigen::Matrix<var, R2, C2>& v2) {
      stan::math::validate_vector(v1,"dot_product");
      stan::math::validate_vector(v2,"dot_product");
      stan::math::validate_matching_sizes(v1,v2,"dot_product");
      return var(new dot_product_vd_vari(v2,v1));
    }
    /**
     * Returns the dot product.
     *
     * @param[in] v1 First array.
     * @param[in] v2 Second array.
     * @param[in] length Length of both arrays.
     * @return Dot product of the arrays.
     */
    inline var dot_product(const var* v1, const var* v2, size_t length) {
      return var(new dot_product_vv_vari(v1, v2, length));
    }
    /**
     * Returns the dot product.
     *
     * @param[in] v1 First array.
     * @param[in] v2 Second array.
     * @param[in] length Length of both arrays.
     * @return Dot product of the arrays.
     */
    inline var dot_product(const var* v1, const double* v2, size_t length) {
      return var(new dot_product_vd_vari(v1, v2, length));
    }
    /**
     * Returns the dot product.
     *
     * @param[in] v1 First array.
     * @param[in] v2 Second array.
     * @param[in] length Length of both arrays.
     * @return Dot product of the arrays.
     */
    inline var dot_product(const double* v1, const var* v2, size_t length) {
      return var(new dot_product_vd_vari(v2, v1, length));
    }
    /**
     * Returns the dot product.
     *
     * @param[in] v1 First vector.
     * @param[in] v2 Second vector.
     * @return Dot product of the vectors.
     * @throw std::domain_error if sizes of v1 and v2 do not match.
     */
    inline var dot_product(const std::vector<var>& v1,
                           const std::vector<var>& v2) {
      stan::math::validate_matching_sizes(v1,v2,"dot_product");
      return var(new dot_product_vv_vari(&v1[0], &v2[0], v1.size()));
    }
    /**
     * Returns the dot product.
     *
     * @param[in] v1 First vector.
     * @param[in] v2 Second vector.
     * @return Dot product of the vectors.
     * @throw std::domain_error if sizes of v1 and v2 do not match.
     */
    inline var dot_product(const std::vector<var>& v1,
                           const std::vector<double>& v2) {
      stan::math::validate_matching_sizes(v1,v2,"dot_product");
      return var(new dot_product_vd_vari(&v1[0], &v2[0], v1.size()));
    }
    /**
     * Returns the dot product.
     *
     * @param[in] v1 First vector.
     * @param[in] v2 Second vector.
     * @return Dot product of the vectors.
     * @throw std::domain_error if sizes of v1 and v2 do not match.
     */
    inline var dot_product(const std::vector<double>& v1,
                           const std::vector<var>& v2) {
      stan::math::validate_matching_sizes(v1,v2,"dot_product");
      return var(new dot_product_vd_vari(&v2[0], &v1[0], v1.size()));
    }

    template<int R1,int C1,int R2, int C2>
    inline Eigen::Matrix<var, 1, C1>
    columns_dot_product(const Eigen::Matrix<var, R1, C1>& v1, 
                        const Eigen::Matrix<var, R2, C2>& v2) {
      stan::math::validate_matching_sizes(v1,v2,"columns_dot_product");
      Eigen::Matrix<var, 1, C1> ret(1,v1.cols());
      for (size_type j = 0; j < v1.cols(); ++j) {
        ret(j) = var(new dot_product_vv_vari(v1.col(j),v2.col(j)));
      }
      return ret;
    }
    
    template<int R1,int C1,int R2, int C2>
    inline Eigen::Matrix<var, 1, C1>
    columns_dot_product(const Eigen::Matrix<var, R1, C1>& v1, 
                        const Eigen::Matrix<double, R2, C2>& v2) {
      stan::math::validate_matching_sizes(v1,v2,"columns_dot_product");
      Eigen::Matrix<var, 1, C1> ret(1,v1.cols());
      for (size_type j = 0; j < v1.cols(); ++j) {
        ret(j) = var(new dot_product_vd_vari(v1.col(j),v2.col(j)));
      }
      return ret;
    }

    template<int R1,int C1,int R2, int C2>
    inline Eigen::Matrix<var, 1, C1>
    columns_dot_product(const Eigen::Matrix<double, R1, C1>& v1, 
                        const Eigen::Matrix<var, R2, C2>& v2) {
      stan::math::validate_matching_sizes(v1,v2,"columns_dot_product");
      Eigen::Matrix<var, 1, C1> ret(1,v1.cols());
      for (size_type j = 0; j < v1.cols(); ++j) {
        ret(j) = var(new dot_product_vd_vari(v2.col(j),v1.col(j)));
      }
      return ret;
    }

    template<int R1,int C1,int R2, int C2>
    inline Eigen::Matrix<var, R1, 1>
    rows_dot_product(const Eigen::Matrix<var, R1, C1>& v1, 
                     const Eigen::Matrix<var, R2, C2>& v2) {
      stan::math::validate_matching_sizes(v1,v2,"rows_dot_product");
      Eigen::Matrix<var, R1, 1> ret(v1.rows(),1);
      for (size_type j = 0; j < v1.rows(); ++j) {
        ret(j) = var(new dot_product_vv_vari(v1.row(j),v2.row(j)));
      }
      return ret;
    }
    
    template<int R1,int C1,int R2, int C2>
    inline Eigen::Matrix<var, R1, 1>
    rows_dot_product(const Eigen::Matrix<var, R1, C1>& v1, 
                     const Eigen::Matrix<double, R2, C2>& v2) {
      stan::math::validate_matching_sizes(v1,v2,"rows_dot_product");
      Eigen::Matrix<var, R1, 1> ret(v1.rows(),1);
      for (size_type j = 0; j < v1.rows(); ++j) {
        ret(j) = var(new dot_product_vd_vari(v1.row(j),v2.row(j)));
      }
      return ret;
    }
    
    template<int R1,int C1,int R2, int C2>
    inline Eigen::Matrix<var, R1, 1>
    rows_dot_product(const Eigen::Matrix<double, R1, C1>& v1, 
                     const Eigen::Matrix<var, R2, C2>& v2) {
      stan::math::validate_matching_sizes(v1,v2,"rows_dot_product");
      Eigen::Matrix<var, R1, 1> ret(v1.rows(),1);
      for (size_type j = 0; j < v1.rows(); ++j) {
        ret(j) = var(new dot_product_vd_vari(v2.row(j),v1.row(j)));
      }
      return ret;
    }
  }
}
#endif
