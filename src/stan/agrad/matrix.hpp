#ifndef __STAN__AGRAD__MATRIX_H__
#define __STAN__AGRAD__MATRIX_H__

// global include
#include <sstream>

#include <Eigen/Dense>

#include <stan/agrad/agrad.hpp>
#include <stan/agrad/special_functions.hpp>
#include <stan/math/matrix.hpp>


namespace Eigen {

  /**
   * Numerical traits template override for Eigen for automatic
   * gradient variables.
   */
  template <> struct NumTraits<stan::agrad::var>
  {
    /**
     * Real-valued variables.
     *
     * Required for numerical traits.
     */
    typedef stan::agrad::var Real;

    /**
     * Non-integer valued variables.
     *
     * Required for numerical traits.
     */
    typedef stan::agrad::var NonInteger;

    /**
     * Nested variables.
     *
     * Required for numerical traits.
     */
    typedef stan::agrad::var Nested;

    /**
     * Return standard library's epsilon for double-precision floating
     * point, <code>std::numeric_limits&lt;double&gt;::epsilon()</code>.
     *
     * @return Same epsilon as a <code>double</code>.
     */
    inline static Real epsilon() { 
      return std::numeric_limits<double>::epsilon(); 
    }

    /**
     * Return dummy precision
     */
    inline static Real dummy_precision() {
      return 1e-12; // copied from NumTraits.h values for double
    }

    /**
     * Return standard library's highest for double-precision floating
     * point, <code>std::numeric_limits&lt;double&gt;::max()</code>.
     *
     * @return Same highest value as a <code>double</code>.
     */
    inline static Real highest() {
      return std::numeric_limits<double>::max();
    }

    /**
     * Return standard library's lowest for double-precision floating
     * point, <code>&#45;std::numeric_limits&lt;double&gt;::max()</code>.
     *
     * @return Same lowest value as a <code>double</code>.
     */    
    inline static Real lowest() {
      return -std::numeric_limits<double>::max();
    }

    /**
     * Properties for automatic differentiation variables
     * read by Eigen matrix library.
     */
    enum {
      IsInteger = 0,
      IsSigned = 1,
      IsComplex = 0,
      RequireInitialization = 0,
      ReadCost = 1,
      AddCost = 1,
      MulCost = 1,
      HasFloatingPoint = 1,
    };
  };

  namespace internal {
    /**
     * Implemented this for printing to stream.
     */
    template<>
    struct significant_decimals_default_impl<stan::agrad::var,false>
    {
      static inline int run()
      {
        using std::ceil;
        return cast<double,int>(ceil(-log(NumTraits<stan::agrad::var>::epsilon().val())
                                     /log(10.0)));
      }
    };

    /**
     * Scalar product traits override for Eigen for automatic
     * gradient variables.
     */
    template <>  
    struct scalar_product_traits<stan::agrad::var,double> {
      typedef stan::agrad::var ReturnType;
    };

    /**
     * Scalar product traits override for Eigen for automatic
     * gradient variables.
     */
    template <>  
    struct scalar_product_traits<double,stan::agrad::var> {
      typedef stan::agrad::var ReturnType;
    };

  }
}

namespace stan {

  namespace agrad {

    /**
     * The type of a matrix holding <code>stan::agrad::var</code>
     * values.
     */
    typedef stan::math::EigenType<var>::matrix matrix_v;

    /**
     * The type of a (column) vector holding <code>stan::agrad::var</code>
     * values.
     */
    typedef stan::math::EigenType<var>::vector vector_v;

    /**
     * The type of a row vector holding <code>stan::agrad::var</code>
     * values.
     */
    typedef stan::math::EigenType<var>::row_vector row_vector_v;

    /**
     * Converts argument to an automatic differentiation variable.
     *
     * Returns a stan::agrad::var variable with the input value.
     *
     * @param[in] x A scalar value
     * @return An automatic differentiation variable with the input value.
     */
    inline var to_var(const double& x) {
      return var(x);
    }
    /**
     * Converts argument to an automatic differentiation variable.
     * 
     * Sets an automatic differentiation variable with the input value.
     *
     * @param[in] x A scalar value
     * @param[out] var_x A reference to an automatic differentiation variable
     *   which will have the value of x.
     */
    inline void to_var(const double& x, var& var_x) {
      var_x = x;
    }
    /**
     * Converts argument to an automatic differentiation variable.
     *
     * Returns a stan::agrad::var variable with the input value.
     *
     * @param[in] x An automatic differentiation variable.
     * @return An automatic differentiation variable with the input value.
     */    
    inline var to_var(const var& x) {
      return x;
    }
    /**
     * Converts argument to an automatic differentiation variable.
     *
     * Sets an automatic differentiation variable with the input value.
     *
     * @param[in] var_in An automatic differentiation variable.
     * @param[out] var_out A reference to an automatic differentiation variable
     *   which will be set to the input value.
     */
    inline void to_var(const var& var_in, var& var_out) {
      var_out = var_in;
    }
    /**
     * Converts argument to an automatic differentiation variable.
     *
     * Returns a stan::agrad::var variable with the input value.
     *
     * @param[in] m A Matrix with scalars
     * @return A Matrix with automatic differentiation variables
     */
    inline matrix_v to_var(const stan::math::matrix_d& m) {
      matrix_v m_v(m.rows(), m.cols());
      for (int j = 0; j < m.cols(); ++j)
        for (int i = 0; i < m.rows(); ++i)
          m_v(i,j) = m(i,j);
      return m_v;
    }
    
    /**
     * Converts argument to an automatic differentiation variable.
     *
     * Sets a Matrix with automatic differentiation variables.
     *
     * @param[in] m A Matrix of scalars.
     * @param[out] m_v A Matrix with automatic differentiation variables
     *    assigned with values of m.
     */
    inline void to_var (const stan::math::matrix_d& m, matrix_v& m_v) {
      m_v.resize(m.rows(), m.cols());
      for (int j = 0; j < m.cols(); ++j)
        for (int i = 0; i < m.rows(); ++i)
          m_v(i,j) = m(i,j);
    }
    /**
     * Converts argument to an automatic differentiation variable.
     *
     * Returns a stan::agrad::var variable with the input value.
     * 
     * @param[in] m A Matrix with automatic differentiation variables.
     * @return A Matrix with automatic differentiation variables.
     */
    inline matrix_v to_var(const matrix_v& m) {
      return m;
    }
    /**
     * Converts argument to an automatic differentiation variable.
     *
     * Sets a Matrix with automatic differentiation variables.
     *
     * @param[in] m_in A Matrix of automatic differentiation variables.
     * @param[out] m_out A Matrix of automatic differentiation variables
     *    assigned with values of m_in.
     */
    inline void to_var(const matrix_v& m_in,
                       matrix_v& m_out) {
      m_out = m_in;
    }
    /**
     * Converts argument to an automatic differentiation variable.
     *
     * Returns a stan::agrad::var variable with the input value.     
     *
     * @param[in] v A Vector of scalars
     * @return A Vector of automatic differentiation variables with
     *   values of v
     */
    inline vector_v to_var(const stan::math::vector_d& v) {
      vector_v v_v(v.size());
      for (int i = 0; i < v.size(); ++i)
        v_v[i] = v[i];
      return v_v;
    }
    /**
     * Converts argument to an automatic differentiation variable.
     *
     * Sets a Vector with automatic differentation variables.
     *
     * @param[in] v A Vector of scalars.
     * @param[out] v_v A Vector of automatic differentation variables with
     *   values of v.
     */
    inline void to_var(const stan::math::vector_d& v,
                       vector_v& v_v) {
      v_v.resize(v.size());
      for (int i = 0; i < v.size(); ++i)
        v_v[i] = v[i];
    }
    /**
     * Converts argument to an automatic differentiation variable.
     *
     * Returns a stan::agrad::var variable with the input value.     
     *
     * @param[in] v A Vector of automatic differentiation variables
     * @return A Vector of automatic differentiation variables with
     *   values of v
     */
    inline vector_v to_var(const vector_v& v) {
      return v;
    }
    /**
     * Converts argument to an automatic differentiation variable.
     *
     * Sets a Vector with automatic differentiation variables
     *
     * @param[in] v_in A Vector of automatic differentiation variables
     * @param[out] v_out A Vector of automatic differentiation variables
     *    with values of v_in
     */
    inline void to_var(const vector_v& v_in,
                       vector_v& v_out) {
      v_out = v_in;
    }
    /**
     * Converts argument to an automatic differentiation variable.
     *
     * Returns a stan::agrad::var variable with the input value.     
     *
     * @param[in] rv A row vector of scalars
     * @return A row vector of automatic differentation variables with 
     *   values of rv.
     */
    inline row_vector_v to_var(const stan::math::row_vector_d& rv) {
      row_vector_v rv_v(rv.size());
      for (int i = 0; i < rv.size(); ++i)
        rv_v[i] = rv[i];
      return rv_v;
    }
    /**
     * Converts argument to an automatic differentiation variable.
     *
     * Sets a row vector with automatic differentiation variables
     *
     * @param[in] rv A row vector of scalars
     * @param[out] rv_v A row vector of automatic differentiation variables
     *   with values set to rv.
     */
    inline void to_var(const stan::math::row_vector_d& rv,
                       row_vector_v& rv_v) {
      rv_v.resize(rv.size());
      for (int i = 0; i < rv.size(); ++i)
        rv_v[i] = rv[i];
    }
    /**
     * Converts argument to an automatic differentiation variable.
     *
     * Returns a stan::agrad::var variable with the input value.     
     *
     * @param[in] rv A row vector with automatic differentiation variables
     * @return A row vector with automatic differentiation variables
     *    with values of rv.
     */
    inline row_vector_v to_var(const row_vector_v& rv) {
      return rv;
    }
    /**
     * Converts argument to an automatic differentiation variable.
     *
     * Sets a row vector with automatic differentiation variables
     *
     * @param[in] rv_in A row vector with automatic differentiation variables
     * @param[out] rv_out A row vector with automatic differentiation variables
     *    with values of rv_in
     */
    inline void to_var(const row_vector_v& rv_in,
                       row_vector_v& rv_out) {
      rv_out = rv_in;
    }

    /**
     * Return number of rows.
     *
     * Returns the number of rows in the specified 
     * column vector.
     *
     * @param v Specified vector.
     * @return Number of rows in the vector.
     */
    inline size_t rows(const vector_v& v) {
      return v.size();
    }
    /**
     * Return number of rows.
     *
     * Returns the number of rows in the specified 
     * row vector.  The return value is always 1.
     *
     * @param rv Specified vector.
     * @return Number of rows in the vector.
     */
    inline size_t rows(const row_vector_v& rv) {
      return 1;
    }
    /**
     * Return number of rows.
     * 
     * Returns the number of rows in the specified matrix.
     * 
     * @param m Specified matrix.
     * @return Number of rows in the vector.
     */
    inline size_t rows(const matrix_v& m) {
      return m.rows();
    }

    /**
     * Return number of columns.
     *
     * Returns the number of columns in the specified
     * column vector.  The return value is always 1.
     *
     * @param v Specified vector.
     * @return Number of columns in the vector.
     */
    inline size_t cols(const vector_v& v) {
      return 1;
    }
    /**
     * Return number of columns.
     *
     * Returns the number of columns in the specified
     * row vector.  
     *
     * @param rv Specified vector.
     * @return Number of columns in the vector.
     */
    inline size_t cols(const row_vector_v& rv) {
      return rv.size();
    }
    /**
     * Return number of columns.
     *
     * Return the number of columns in the specified matrix.
     *
     * @param m Specified matrix.
     * @return Number of columns in the matrix.
     */
    inline size_t cols(const matrix_v& m) {
      return m.cols();
    }

    // scalar returns

    /**
     * Determinant of the matrix.
     *
     * Returns the determinant of the specified
     * square matrix.
     *
     * @param m Specified matrix.
     * @return Determinant of the matrix.
     * @throw std::domain_error if m is not a square matrix
     */
    var determinant(const Eigen::Matrix<var, Eigen::Dynamic, Eigen::Dynamic>& m);
    
    namespace {
      class dot_self_vari : public vari {
      protected:
        vari** v_;
        size_t size_;
      public:
        template <int R, int C>
        dot_self_vari(const Eigen::Matrix<var,R,C>& v) :
          vari(var_dot_self(v)), size_(v.size()) {
          v_ = (vari**) memalloc_.alloc(size_ * sizeof(vari*));
          for (size_t i = 0; i < size_; ++i)
            v_[i] = v(i).vi_;
        }
        inline static double square(double x) { return x * x; }
        template <int R, int C>
        inline static double var_dot_self(const Eigen::Matrix<var,R,C> &v) {
          double sum = 0.0;
          for (int i = 0; i < v.size(); ++i)
            sum += square(v(i).vi_->val_);
          return sum;
        }
        void chain() {
          for (size_t i = 0; i < size_; ++i) 
            v_[i]->adj_ += adj_ * 2.0 * v_[i]->val_;
        }
      };

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
        template<int R1,int C1,int R2,int C2>
        inline static double var_dot(const Eigen::Matrix<var,R1,C1> &v1,
                                     const Eigen::Matrix<var,R2,C2> &v2) {
          double result = 0;
          for (int i = 0; i < v1.size(); i++)
            result += v1[i].vi_->val_ * v2[i].vi_->val_;
          return result;
        }
      public:
        dot_product_vv_vari(const var* v1, const var* v2, size_t length) : 
          vari(var_dot(v1, v2, length)), length_(length) {
          v1_ = (vari**)memalloc_.alloc(2*length_*sizeof(vari*));
          v2_ = v1_ + length_;
          for (size_t i = 0; i < length_; i++)
            v1_[i] = v1[i].vi_;
          for (size_t i = 0; i < length_; i++)
            v2_[i] = v2[i].vi_;
        }
        template<int R1,int C1,int R2,int C2>
        dot_product_vv_vari(const Eigen::Matrix<var,R1,C1> &v1,
                            const Eigen::Matrix<var,R2,C2> &v2) : 
          vari(var_dot(v1, v2)), length_(v1.size()) {
          v1_ = (vari**)memalloc_.alloc(2*length_*sizeof(vari*));
          v2_ = v1_ + length_;
          for (size_t i = 0; i < length_; i++)
            v1_[i] = v1[i].vi_;
          for (size_t i = 0; i < length_; i++)
            v2_[i] = v2[i].vi_;
        }
        void chain() {
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
        template<int R1,int C1,int R2,int C2>
        inline static double var_dot(const Eigen::Matrix<var,R1,C1> &v1,
                                     const Eigen::Matrix<double,R2,C2> &v2) {
          double result = 0;
          for (int i = 0; i < v1.size(); i++)
            result += v1[i].vi_->val_ * v2[i];
          return result;
        }
      public:
        dot_product_vd_vari(const var* v1, const double* v2, size_t length) : 
          vari(var_dot(v1, v2, length)), length_(length) {
          v1_ = (vari**)memalloc_.alloc(length_*sizeof(vari*));
          v2_ = (double*)memalloc_.alloc(length_*sizeof(double));
          for (size_t i = 0; i < length_; i++)
            v1_[i] = v1[i].vi_;
          for (size_t i = 0; i < length_; i++)
            v2_[i] = v2[i];
        }
        template<int R1,int C1,int R2,int C2>
        dot_product_vd_vari(const Eigen::Matrix<var,R1,C1> &v1,
                            const Eigen::Matrix<double,R2,C2> &v2) : 
          vari(var_dot(v1, v2)), length_(v1.size()) {
          v1_ = (vari**)memalloc_.alloc(length_*sizeof(vari*));
          v2_ = (double*)memalloc_.alloc(length_*sizeof(double));
          for (size_t i = 0; i < length_; i++)
            v1_[i] = v1[i].vi_;
          for (size_t i = 0; i < length_; i++)
            v2_[i] = v2[i];
        }
        void chain() {
          for (size_t i = 0; i < length_; i++) {
            v1_[i]->adj_ += adj_ * v2_[i];
          }
        }
      };

      // FIXME: untested
      class gevv_vvv_vari : public stan::agrad::vari {
      protected:
        stan::agrad::vari* alpha_;
        stan::agrad::vari** v1_;
        stan::agrad::vari** v2_;
        size_t length_;
        inline static double eval_gevv(const stan::agrad::var* alpha,
                                       const stan::agrad::var* v1, int stride1,
                                       const stan::agrad::var* v2, int stride2,
                                       size_t length) {
          double result = 0;
          for (size_t i = 0; i < length; i++)
            result += alpha->vi_->val_ * v1[i*stride1].vi_->val_ * v2[i*stride2].vi_->val_;
          return result;
        }
      public:
        gevv_vvv_vari(const stan::agrad::var* alpha, 
                      const stan::agrad::var* v1, int stride1, 
                      const stan::agrad::var* v2, int stride2, size_t length) : 
          vari(eval_gevv(alpha, v1, stride1, v2, stride2, length)), length_(length) {
          alpha_ = alpha->vi_;
          v1_ = (stan::agrad::vari**)stan::agrad::memalloc_.alloc(2*length_*sizeof(stan::agrad::vari*));
          v2_ = v1_ + length_;
          for (size_t i = 0; i < length_; i++)
            v1_[i] = v1[i*stride1].vi_;
          for (size_t i = 0; i < length_; i++)
            v2_[i] = v2[i*stride2].vi_;
        }
        void chain() {
          for (size_t i = 0; i < length_; i++) {
            v1_[i]->adj_ += adj_ * v2_[i]->val_ * alpha_->val_;
            v2_[i]->adj_ += adj_ * v1_[i]->val_ * alpha_->val_;
            alpha_->adj_ += adj_ * v1_[i]->val_ * v2_[i]->val_;
          }
        }
      };
    }

    /**
     * Returns the dot product of a vector with itself.
     *
     * @param[in] v Vector.
     * @return Dot product of the vector with itself.
     * @tparam R number of rows or <code>Eigen::Dynamic</code> for dynamic; one of R or C must be 1
     * @tparam C number of rows or <code>Eigen::Dyanmic</code> for dynamic; one of R or C must be 1
     */
    template<int R, int C>
    inline var dot_self(const Eigen::Matrix<var, R, C>& v) {
      if (v.rows() != 1 && v.cols() != 1)
        throw std::invalid_argument("v must be a vector");
      return var(new dot_self_vari(v));
    }
    
    /**
     * Returns the dot product.
     *
     * @param[in] v1 First column vector.
     * @param[in] v2 Second column vector.
     * @return Dot product of the vectors.
     * @throw std::invalid_argument if length of v1 is not equal to length of v2.
     */
    template<int R1,int C1,int R2, int C2>
    inline var dot_product(const Eigen::Matrix<var, R1, C1>& v1, 
                           const Eigen::Matrix<var, R2, C2>& v2) {
      if (v1.rows() != 1 && v1.cols() != 1)
        throw std::invalid_argument("v1 must be a vector");
      if (v2.rows() != 1 && v2.cols() != 1)
        throw std::invalid_argument("v2 must be a vector");
      if (v1.size() != v2.size())
        throw std::invalid_argument("v1.size() must equal v2.size()");
      return var(new dot_product_vv_vari(v1,v2));
    }
    /**
     * Returns the dot product.
     *
     * @param[in] v1 First column vector.
     * @param[in] v2 Second column vector.
     * @return Dot product of the vectors.
     * @throw std::invalid_argument if length of v1 is not equal to length of v2
     * or either v1 or v2 are not vectors.
     */
    template<int R1,int C1,int R2, int C2>
    inline var dot_product(const Eigen::Matrix<var, R1, C1>& v1, 
                           const Eigen::Matrix<double, R2, C2>& v2) {
      if (v1.rows() != 1 && v1.cols() != 1)
        throw std::invalid_argument("v1 must be a vector");
      if (v2.rows() != 1 && v2.cols() != 1)
        throw std::invalid_argument("v2 must be a vector");
      if (v1.size() != v2.size())
        throw std::invalid_argument("v1.size() must equal v2.size()");
      return var(new dot_product_vd_vari(v1,v2));
    }
    /**
     * Returns the dot product.
     *
     * @param[in] v1 First column vector.
     * @param[in] v2 Second column vector.
     * @return Dot product of the vectors.
     * @throw std::invalid_argument if length of v1 is not equal to length of v2
     * or either v1 or v2 are not vectors.
     */
    template<int R1,int C1,int R2, int C2>
    inline var dot_product(const Eigen::Matrix<double, R1, C1>& v1, 
                           const Eigen::Matrix<var, R2, C2>& v2) {
      if (v1.rows() != 1 && v1.cols() != 1)
        throw std::invalid_argument("v1 must be a vector");
      if (v2.rows() != 1 && v2.cols() != 1)
        throw std::invalid_argument("v2 must be a vector");
      if (v1.size() != v2.size())
        throw std::invalid_argument("v1.size() must equal v2.size()");
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
     * @throw std::invalid_argument if sizes of v1 and v2 do not match.
     */
    inline var dot_product(const std::vector<var>& v1,
                           const std::vector<var>& v2) {
      if (v1.size() != v2.size())
        throw std::invalid_argument("v1.size() must equal v2.size()");
      return var(new dot_product_vv_vari(&v1[0], &v2[0], v1.size()));
    }
    /**
     * Returns the dot product.
     *
     * @param[in] v1 First vector.
     * @param[in] v2 Second vector.
     * @return Dot product of the vectors.
     * @throw std::invalid_argument if sizes of v1 and v2 do not match.
     */
    inline var dot_product(const std::vector<var>& v1,
                           const std::vector<double>& v2) {
      if (v1.size() != v2.size())
        throw std::invalid_argument("v1.size() must equal v2.size()");
      return var(new dot_product_vd_vari(&v1[0], &v2[0], v1.size()));
    }
    /**
     * Returns the dot product.
     *
     * @param[in] v1 First vector.
     * @param[in] v2 Second vector.
     * @return Dot product of the vectors.
     * @throw std::invalid_argument if sizes of v1 and v2 do not match.
     */
    inline var dot_product(const std::vector<double>& v1,
                           const std::vector<var>& v2) {
      if (v1.size() != v2.size())
        throw std::invalid_argument("v1.size() must equal v2.size()");
      return var(new dot_product_vd_vari(&v2[0], &v1[0], v1.size()));
    }

    /**
     * Returns the minimum coefficient in the specified
     * column vector.
     *
     * @param[in] v Specified vector.
     * @return Minimum coefficient value in the vector.
     * @throw std::domain_error if v has no elements
     */
    template <typename T>
    inline var min(const Eigen::Matrix<T, Eigen::Dynamic, 1>& v) {
      if (v.size() == 0) 
        throw std::domain_error ("v.size() == 0");
      return to_var(v.minCoeff());
    }
    /**
     * Returns the minimum coefficient in the specified
     * row vector.
     *
     * @param[in] rv Specified vector.
     * @return Minimum coefficient value in the vector.
     * @throw std::domain_error if rv has no elements
     */
    template <typename T>
    inline var min(const Eigen::Matrix<T, 1, Eigen::Dynamic>& rv) {
      if (rv.size() == 0) 
        throw std::domain_error ("rv.size() == 0");
      return to_var(rv.minCoeff());
    }
    /**
     * Returns the minimum coefficient in the specified
     * matrix.
     *
     * @param[in] m Specified matrix.
     * @return Minimum coefficient value in the matrix.
     * @throw std::domain_error if m has no elements
     */
    template <typename T>
    inline var min(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& m) {
      if (m.size() == 0) 
        throw std::domain_error ("m.size() == 0");
      return to_var(m.minCoeff());
    }

    /**
     * Returns the maximum coefficient in the specified
     * column vector.
     *
     * @param[in] v Specified vector.
     * @return Maximum coefficient value in the vector.
     * @throw std::domain_error if v has no elements
     */
    template <typename T>
    inline var max(const Eigen::Matrix<T, Eigen::Dynamic, 1>& v) {
      if (v.size() == 0) 
        throw std::domain_error ("v.size() == 0");
      return to_var(v.maxCoeff());
    }
    /**
     * Returns the maximum coefficient in the specified
     * row vector.
     * @param[in] rv Specified vector.
     * @return Maximum coefficient value in the vector.
     * @throw std::domain_error if rv has no elements
     */
    template <typename T>
    inline var max(const Eigen::Matrix<T, 1, Eigen::Dynamic>& rv) {
      if (rv.size() == 0) 
        throw std::domain_error ("rv.size() == 0");
      return to_var(rv.maxCoeff());
    }
    /**
     * Returns the maximum coefficient in the specified
     * matrix.
     * @param[in] m Specified matrix.
     * @return Maximum coefficient value in the matrix.
     * @throw std::domain_error if m has no elements
     */
    template <typename T>
    inline var max(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& m) {
      if (m.size() == 0) 
        throw std::domain_error ("m.size() == 0");
      return to_var(m.maxCoeff());
    }

    /**
     * Returns the sample mean (i.e., average) of the coefficients
     * in the specified column vector.
     * @param[in] v Specified vector.
     * @return Sample mean of vector coefficients.
     * @throw std::domain_error if v has no elements
     */
    template <typename T>
    inline var mean(const Eigen::Matrix<T, Eigen::Dynamic, 1>& v) {
      if (v.size() == 0) 
        throw std::domain_error ("v.size() == 0");
      return to_var(v.mean());
    }
    /**
     * Returns the sample mean (i.e., average) of the coefficients
     * in the specified row vector.
     * @param[in] rv Specified vector.
     * @return Sample mean of vector coefficients.
     * @throw std::domain_error if rv has no elements
     */
    template <typename T>
    inline var mean(const Eigen::Matrix<T, 1, Eigen::Dynamic>& rv) {
      if (rv.size() == 0) 
        throw std::domain_error ("rv.size() == 0");
      return to_var(rv.mean());
    }
    /**
     * Returns the sample mean (i.e., average) of the coefficients
     * in the specified matrix.
     * @param[in] m Specified matrix.
     * @return Sample mean of matrix coefficients.
     * @throw std::domain_error if m has no elements
     */
    template <typename T>
    inline var mean(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& m) {
      if (m.size() == 0) 
        throw std::domain_error ("m.size() == 0");
      return to_var(m.mean());
    }

    /**
     * Returns the sample variance (divide by length - 1) of the
     * coefficients in the specified column vector.
     * @param[in] v Specified vector.
     * @return Sample variance of vector. If there is only one element, returns 0.0
     * @throw std::domain_error if v has no elements
     */
    template <typename T>
    inline var variance(const Eigen::Matrix<T, Eigen::Dynamic, 1>& v) {
      if (v.size() == 0) 
        throw std::domain_error ("v.size() == 1");
      if (v.size() == 1) 
        return to_var(0.0);
      T mean = v.mean();
      T sum_sq_diff = 0;
      // FIXME: redefine in terms of vectorized ops
      // FIXME: should we use Welford's algorithm for numeric stability?
      // (v.array() - mean).square().sum() / (v.size() - 1);
      for (int i = 0; i < v.size(); ++i) {
        T diff = v[i] - mean;
        sum_sq_diff += diff * diff;
      }
      return to_var(sum_sq_diff / (v.size() - 1));
    }
    /**
     * Returns the sample variance (divide by length - 1) of the
     * coefficients in the specified row vector.
     * @param[in] rv Specified vector.
     * @return Sample variance of vector. If there is only one element, returns 0.0
     * @throw std::domain_error if rv has no elements
     */
    template <typename T>
    inline var variance(const Eigen::Matrix<T, 1, Eigen::Dynamic>& rv) {
      if (rv.size() == 0) 
        throw std::domain_error ("rv.size() == 0");
      if (rv.size() == 1)
        return to_var(0.0);
      T mean = rv.mean();
      T sum_sq_diff = 0;
      for (int i = 0; i < rv.size(); ++i) {
        T diff = rv[i] - mean;
        sum_sq_diff += diff * diff;
      }
      return to_var(sum_sq_diff / (rv.size() - 1));
    }
    /**
     * Returns the sample variance (divide by length - 1) of the
     * coefficients in the specified matrix.
     * @param[in] m Specified matrix.
     * @return Sample variance of vector. If there is only one element, returns 0.0
     * @throw std::domain_error if v has no elements
     */
    template <typename T>
    inline var variance(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& m) {
      if (m.size() == 0) 
        throw std::domain_error ("m.size() == 0");
      if (m.size() == 1) 
        return to_var(0.0);
      T mean = m.mean();
      T sum_sq_diff = 0;
      for (int j = 0; j < m.cols(); ++j) { 
        for (int i = 0; i < m.rows(); ++i) {
          T diff = m(i,j) - mean;
          sum_sq_diff += diff * diff;
        }
      }
      return to_var(sum_sq_diff / (m.size() - 1));
    }

    /**
     * Returns the unbiased sample standard deviation of the
     * coefficients in the specified column vector.
     * @param[in] v Specified vector.
     * @return Sample standard deviation of vector. If there is only one element, 
     * returns 0.0
     * @throw std::domain_error if v has no elements
     */
    template <typename T>
    inline var sd(const Eigen::Matrix<T, Eigen::Dynamic, 1>& v) {
      if (v.size() == 0) 
        throw std::domain_error ("v.size() == 0");
      return to_var(sqrt(variance(v)));
    }
    /**
     * Returns the unbiased sample standard deviation of the
     * coefficients in the specified row vector.
     * @param[in] rv Specified vector.
     * @return Sample standard deviation of vector. If there is only one element, 
     * returns 0.0
     * @throw std::domain_error if rv has no elements
     */
    template <typename T>
    inline var sd(const Eigen::Matrix<T, 1, Eigen::Dynamic>& rv) {
      if (rv.size() == 0) 
        throw std::domain_error ("rv.size() <= 1");
      return to_var(sqrt(variance(rv)));
    }
    /**
     * Returns the unbiased sample standard deviation of the
     * coefficients in the specified matrix.
     * @param[in] m Specified matrix.
     * @return Sample standard deviation of a matrix. If there is only one element, 
     * returns 0.0
     * @throw std::domain_error if m has no elements
     */
    template <typename T>
    inline var sd(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& m) {
      if (m.size() == 0) 
        throw std::domain_error ("m.size() == 0");
      return to_var(sqrt(variance(m)));
    }

    /**
     * Returns the sum of the coefficients of the specified
     * column vector.
     * @param[in] v Specified vector.
     * @return Sum of coefficients of vector.
     */
    template <typename T>
    inline var sum(const Eigen::Matrix<T, Eigen::Dynamic, 1>& v) {
      return to_var(v.sum());
    }
    /**
     * Returns the sum of the coefficients of the specified
     * row vector.
     * @param[in] rv Specified vector.
     * @return Sum of coefficients of vector.
     */
    template <typename T>
    inline var sum(const Eigen::Matrix<T, 1, Eigen::Dynamic>& rv) {
      return to_var(rv.sum());
    }
    /**
     * Returns the sum of the coefficients of the specified
     * matrix
     * @param[in] m Specified matrix.
     * @return Sum of coefficients of matrix.
     */
    template <typename T>
    inline var sum(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& m) {
      return to_var(m.sum());
    }

    /**
     * Returns the product of the coefficients of the specified
     * column vector.
     * @param[in] v Specified vector.
     * @return Product of coefficients of vector.
     */
    inline var prod(const vector_v& v) {
      return v.prod();
    }
    /**
     * Returns the product of the coefficients of the specified
     * row vector.
     * @param[in] rv Specified vector.
     * @return Product of coefficients of vector.
     */
    inline var prod(const row_vector_v& rv) {
      return rv.prod();
    }
    /**
     * Returns the product of the coefficients of the specified
     * matrix.
     * @param[in] m Specified matrix.
     * @return Product of coefficients of matrix.
     */
    inline var prod(const matrix_v& m) {
      return m.prod();
    }

    /**
     * Returns the trace of the specified matrix.  The trace
     * is defined as the sum of the elements on the diagonal.
     * The matrix is not required to be square.
     *
     * @param[in] m Specified matrix.
     * @return Trace of the matrix.
     */
    inline var trace(const matrix_v& m) {
      return m.trace();
    }
    
    /**
     * Return the element-wise logarithm of the matrix or vector.
     *
     * @param[in] m The matrix or vector.
     * @return ret(i,j) = log(m(i,j))
     */
    template<typename TM, int Rows, int Cols>
    inline Eigen::Matrix<var,Rows,Cols> log(const Eigen::Matrix<TM,Rows,Cols>& m) {
      return to_var(m).array().log().matrix();
    }
   
    /**
     * Return the element-wise exponentiation of the matrix or vector.
     *
     * @param[in] m The matrix or vector.
     * @return ret(i,j) = exp(m(i,j))
     */
    template<typename TM, int Rows, int Cols>
    inline Eigen::Matrix<var,Rows,Cols> exp(const Eigen::Matrix<TM,Rows,Cols>& m) {
      return to_var(m).array().exp().matrix();
    }
     
    /**
     * Return the sum of the specified column vectors.
     * The two vectors must have the same size.
     *
     * @param[in] v1 First vector.
     * @param[in] v2 Second vector.
     * @return Sum of the two vectors.
     * @throw std::invalid_argument if size of v1 is not equal to size of v2.
     */
    template <typename T1, typename T2>
    inline Eigen::Matrix<var, Eigen::Dynamic, 1> 
    add(const Eigen::Matrix<T1, Eigen::Dynamic, 1>& v1, 
        const Eigen::Matrix<T2, Eigen::Dynamic, 1>& v2) {
      if (v1.size() != v2.size())
        throw std::invalid_argument("v1.size() must equal v2.size()");
      return to_var(v1) + to_var(v2);
    }
    /**
     * Return the sum of the specified row vectors.  The
     * two vectors must have the same size.
     *
     * @param[in] rv1 First vector.
     * @param[in] rv2 Second vector.
     * @return Sum of the two vectors.
     * @throw std::invalid_argument if size of rv1 is not equal to size of rv2.
     */
    template <typename T1, typename T2>
    inline Eigen::Matrix<var, 1, Eigen::Dynamic>
    add(const Eigen::Matrix<T1, 1, Eigen::Dynamic>& rv1, 
        const Eigen::Matrix<T2, 1, Eigen::Dynamic>& rv2) {
      if (rv1.size() != rv2.size())
        throw std::invalid_argument("rv1.size() must equal rv2.size()");
      return to_var(rv1) + to_var(rv2);
    }

    /**
     * Return the sum of the specified matrices.  The two matrices
     * must have the same dimensions.
     *
     * @param[in] m1 First matrix.
     * @param[in] m2 Second matrix.
     * @return Sum of the two vectors.
     * @throw std::invalid_argument if dimension of m1 and m2 do not match.
     */
    template <typename T1, typename T2>
    inline Eigen::Matrix<var, Eigen::Dynamic, Eigen::Dynamic>
    add(const Eigen::Matrix<T1, Eigen::Dynamic, Eigen::Dynamic>& m1, 
        const Eigen::Matrix<T2, Eigen::Dynamic, Eigen::Dynamic>& m2) {
      if (m1.rows() != m2.rows() || m1.cols() != m2.cols())
        throw std::invalid_argument("m1 dimensions must match m2 dimensions");
      return to_var(m1) + to_var(m2);
    }

    /**
     * Return the sum of a matrix or vector and a scalar.
     * @param[in] m Matrix or vector.
     * @param[in] c Scalar.
     * @return Matrix or Vector plus the scalar.
     */
    template <typename T1, typename T2, int Rows, int Cols>
    inline Eigen::Matrix<var, Rows, Cols>
    add(const Eigen::Matrix<T1, Rows, Cols>& m,
        const T2& c) {
      return (to_var(m).array() + c).matrix();
    }

    /**
     * Return the sum of a scalar and a matrix or vector.
     * @param[in] c Scalar.
     * @param[in] m Matrix or vector.
     * @return Scalar plus vector.
     */
    template <typename T1, typename T2, int Rows, int Cols>
    inline Eigen::Matrix<var, Rows, Cols>
    add(const T1& c,
        const Eigen::Matrix<T2, Rows, Cols>& m) {
      return (c + to_var(m).array()).matrix();
    }

    /**
     * Return the difference between a matrix or vector  and a scalar.
     * @param[in] m Matrix or vector.
     * @param[in] c Scalar.
     * @return Vector minus the scalar.
     */
    template <typename T1, typename T2, int Rows, int Cols>
    inline Eigen::Matrix<var, Rows, Cols>
    subtract(const Eigen::Matrix<T1, Rows, Cols>& m,
             const T2& c) {
      return (to_var(m).array() - c).matrix();
    }
    /**
     * Return the difference between a scalar and a matrix or vector.
     * @param[in] c Scalar.
     * @param[in] m Matrix or vector.
     * @return Scalar minus vector.
     */
    template <typename T1, typename T2, int Rows, int Cols>
    inline Eigen::Matrix<var, Rows, Cols>
    subtract(const T1& c,
             const Eigen::Matrix<T2, Rows, Cols>& m) {
      return (c - to_var(m).array()).matrix();
    }
    /**
     * Return the difference between the first specified column vector
     * and the second.  The two vectors must have the same size.
     * @param[in] v1 First vector.
     * @param[in] v2 Second vector.
     * @return First vector minus the second vector.
     * @throw std::invalid_argument if size of v1 does not match size of v2.
     */
    template <typename T1, typename T2>
    inline Eigen::Matrix<var, Eigen::Dynamic, 1> 
    subtract(const Eigen::Matrix<T1, Eigen::Dynamic, 1>& v1, 
             const Eigen::Matrix<T2, Eigen::Dynamic, 1>& v2) {
      if (v1.size() != v2.size())
        throw std::invalid_argument("v1.size() must equal v2.size()");
      return to_var(v1) - to_var(v2);
    }
    /**
     * Return the difference between the first specified row vector
     * and the second.  The two vectors must have the same size.
     * @param[in] rv1 First vector.
     * @param[in] rv2 Second vector.
     * @return First vector minus the second vector.
     * @throw std::invalid_argument if size of rv1 does not match size of rv2.
     */
    template <typename T1, typename T2>
    inline Eigen::Matrix<var, 1, Eigen::Dynamic> 
    subtract(const Eigen::Matrix<T1, 1, Eigen::Dynamic>& rv1, 
             const Eigen::Matrix<T2, 1, Eigen::Dynamic>& rv2) {
      if (rv1.size() != rv2.size())
        throw std::invalid_argument("rv1.size() must equal rv2.size()");
      return to_var(rv1) - to_var(rv2);
    }

    /**
     * Return the difference between the first specified matrix and
     * the second.  The two matrices must have the same dimensions.
     * @param[in] m1 First matrix.
     * @param[in] m2 Second matrix.
     * @return First matrix minus the second matrix.
     * @throw std::invalid_argument if dimension of m1 and m2 do not match.
     */
    template <typename T1, typename T2>
    inline Eigen::Matrix<var, Eigen::Dynamic, Eigen::Dynamic> 
    subtract(const Eigen::Matrix<T1, Eigen::Dynamic, Eigen::Dynamic>& m1, 
             const Eigen::Matrix<T2, Eigen::Dynamic, Eigen::Dynamic>& m2) {
      if (m1.rows() != m2.rows() || m1.cols() != m2.cols())
        throw std::invalid_argument("m1 dimensions must match m2 dimensions");
      return to_var(m1) - to_var(m2);
    }


    /**
     * Return the negation of the specified variable.  The result
     * is the same as multiplying by the scalar <code>-1</code>.
     * @param[in] v Specified variable.  
     * @return The negation of the variable.
     */
    template <typename T>
    inline var minus(const T& v) {
      return -to_var(v);
    }
    /**
     * Return the negation of the specified column vector.  The result
     * is the same as multiplying by the scalar <code>-1</code>.
     * @param[in] v Specified vector.  
     * @return The negation of the vector.
     */
    template <typename T>
    inline vector_v minus(const Eigen::Matrix<T, Eigen::Dynamic, 1>& v) {
      return -to_var(v);
    }
    /**
     * Return the negation of the specified row vector.  The result is
     * the same as multiplying by the scalar <code>-1</code>.
     * @param[in] rv Specified vector.
     * @return The negation of the vector.
     */
    template <typename T>
    inline row_vector_v minus(const Eigen::Matrix<T, 1, Eigen::Dynamic>& rv) {
      return -to_var(rv);
    }
    /**
     * Return the negation of the specified matrix.  The result is the same
     * as multiplying by the scalar <code>-1</code>.
     * @param[in] m Specified matrix.
     * @return The negation of the matrix.
     */
    template <typename T>
    inline matrix_v minus(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& m) {
      return -to_var(m);
    }

    /**
     * Return the division of the first scalar by
     * the second scalar.
     * @param[in] v Specified vector.
     * @param[in] c Specified scalar.
     * @return Vector divided by the scalar.
     */
    template <typename T1, typename T2>
    inline var divide(const T1& v, const T2& c) {
      return to_var(v) / to_var(c);
    }

    /**
     * Return the division of the specified column vector by
     * the specified scalar.
     * @param[in] v Specified vector.
     * @param[in] c Specified scalar.
     * @return Vector divided by the scalar.
     */
    template <typename T1, typename T2>
    inline vector_v divide(const Eigen::Matrix<T1, Eigen::Dynamic, 1>& v, const T2& c) {
      return to_var(v) / to_var(c);
    }

    /**
     * Return the division of the specified row vector by
     * the specified scalar.
     * @param[in] rv Specified vector.
     * @param[in] c Specified scalar.
     * @return Vector divided by the scalar.
     */
    template <typename T1, typename T2>
    inline row_vector_v divide(const Eigen::Matrix<T1, 1, Eigen::Dynamic>& rv,
                               const T2& c) {
      return to_var(rv) / to_var(c);
    }
    /**
     * Return the division of the specified matrix by the specified
     * scalar.
     * @param[in] m Specified matrix.
     * @param[in] c Specified scalar.
     * @return Matrix divided by the scalar.
     */
    template <typename T1, typename T2>
    inline matrix_v divide(const Eigen::Matrix<T1, Eigen::Dynamic, Eigen::Dynamic>& m, 
                           const T2& c) {
      return to_var(m) / to_var(c);
    }

    /**
     * Return the elementwise product of the specified matrices.
     * @param[in] m1 First matrix.
     * @param[in] m2 Second matrix.
     * @return Elementwise product of the matrices.
     */
    template <typename T1, typename T2>
    inline Eigen::Matrix<stan::agrad::var,Eigen::Dynamic,Eigen::Dynamic>
    elt_multiply(const Eigen::Matrix<T1,Eigen::Dynamic,Eigen::Dynamic>& m1,
                 const Eigen::Matrix<T2,Eigen::Dynamic,Eigen::Dynamic>& m2) {
      Eigen::Matrix<stan::agrad::var,Eigen::Dynamic, Eigen::Dynamic> 
        result(m1.rows(),m1.cols());
      for (int j = 0; j < m1.cols(); ++j)
        for (int i = 0; i < m1.rows(); ++i)
          result(i,j) = m1(i,j) * m2(i,j);
      return result;
    }
    /**
     * Return the elementwise product of the specified vectors.
     * @param[in] v1 First vector.
     * @param[in] v2 Second vector.
     * @return Elementwise product of the vectors.
     */
    template <typename T1, typename T2>
    inline Eigen::Matrix<stan::agrad::var,Eigen::Dynamic,1>
    elt_multiply(const Eigen::Matrix<T1,Eigen::Dynamic,1>& v1,
                 const Eigen::Matrix<T2,Eigen::Dynamic,1>& v2) {
      Eigen::Matrix<stan::agrad::var,Eigen::Dynamic,1> result(v1.size());
      for (int i = 0; i < v1.size(); ++i)
        result(i) = v1(i) * v2(i);
      return result;
    }
    /**
     * Return the elementwise product of the specified row vectors.
     * @param[in] v1 First row vector.
     * @param[in] v2 Second row vector.
     * @return Elementwise product of the row vectors.
     */
    template <typename T1, typename T2>
    inline Eigen::Matrix<stan::agrad::var,1,Eigen::Dynamic>
    elt_multiply(const Eigen::Matrix<T1,1,Eigen::Dynamic>& v1,
                 const Eigen::Matrix<T2,1,Eigen::Dynamic>& v2) {
      Eigen::Matrix<stan::agrad::var,1,Eigen::Dynamic> result(v1.size());
      for (int i = 0; i < v1.size(); ++i)
        result(i) = v1(i) * v2(i);
      return result;
    }
                   

    /**
     * Return the elementwise division of the specified matrices.
     * @param[in] m1 First matrix.
     * @param[in] m2 Second matrix.
     * @return Elementwise division of the matrices.
     */
    template <typename T1, typename T2>
    inline Eigen::Matrix<stan::agrad::var,Eigen::Dynamic,Eigen::Dynamic>
    elt_divide(const Eigen::Matrix<T1,Eigen::Dynamic,Eigen::Dynamic>& m1,
               const Eigen::Matrix<T2,Eigen::Dynamic,Eigen::Dynamic>& m2) {
      Eigen::Matrix<stan::agrad::var,Eigen::Dynamic, Eigen::Dynamic> 
        result(m1.rows(),m1.cols());
      for (int j = 0; j < m1.cols(); ++j)
        for (int i = 0; i < m1.rows(); ++i)
          result(i,j) = m1(i,j) / m2(i,j);
      return result;
    }
    /**
     * Return the elementwise division of the specified vectors.
     * @param[in] v1 First vector.
     * @param[in] v2 Second vector.
     * @return Elementwise division of the vectors.
     */
    template <typename T1, typename T2>
    inline Eigen::Matrix<stan::agrad::var,Eigen::Dynamic,1>
    elt_divide(const Eigen::Matrix<T1,Eigen::Dynamic,1>& v1,
               const Eigen::Matrix<T2,Eigen::Dynamic,1>& v2) {
      Eigen::Matrix<stan::agrad::var,Eigen::Dynamic,1> result(v1.size());
      for (int i = 0; i < v1.size(); ++i)
        result(i) = v1(i) / v2(i);
      return result;
    }
    /**
     * Return the elementwise division of the specified row vectors.
     * @param[in] v1 First row vector.
     * @param[in] v2 Second row vector.
     * @return Elementwise division of the row vectors.
     */
    template <typename T1, typename T2>
    inline Eigen::Matrix<stan::agrad::var,1,Eigen::Dynamic>
    elt_divide(const Eigen::Matrix<T1,1,Eigen::Dynamic>& v1,
               const Eigen::Matrix<T2,1,Eigen::Dynamic>& v2) {
      Eigen::Matrix<stan::agrad::var,1,Eigen::Dynamic> result(v1.size());
      for (int i = 0; i < v1.size(); ++i)
        result(i) = v1(i) / v2(i);
      return result;
    }

    /**
     * Return the product of two scalars.
     * @param[in] v First scalar.
     * @param[in] c Specified scalar.
     * @return Product of vector and scalar.
     */
    template <typename T1, typename T2>
    inline var multiply(const T1& v, const T2& c) {
      return to_var(v) * to_var(c);
    }

    /**
     * Return the product of scalar and matrix.
     * @param[in] c Specified scalar.
     * @param[in] m Matrix.
     * @return Product of scalar and matrix.
     */
    template<typename T1,typename T2,int R2,int C2>
    inline Eigen::Matrix<var,R2,C2> multiply(const T1& c, 
                                             const Eigen::Matrix<T2, R2, C2>& m) {
      return to_var(m) * to_var(c);
    }

    /**
     * Return the product of scalar and matrix.
     * @param[in] m Matrix.
     * @param[in] c Specified scalar.
     * @return Product of scalar and matrix.
     */
    template<typename T1,int R1,int C1,typename T2>
    inline Eigen::Matrix<var,R1,C1> multiply(const Eigen::Matrix<T1, R1, C1>& m, 
                                             const T2& c) {
      return to_var(m) * to_var(c);
    }
    
    /**
     * Return the product of the specified matrices.  The number of
     * columns in the first matrix must be the same as the number of rows
     * in the second matrix.
     * @param[in] m1 First matrix.
     * @param[in] m2 Second matrix.
     * @return The product of the first and second matrices.
     * @throw std::invalid_argument if the number of columns of m1 does not match
     *   the number of rows of m2.
     */
    template<int R1,int C1,int R2,int C2>
    inline Eigen::Matrix<var,R1,C2> multiply(const Eigen::Matrix<var,R1,C1>& m1,
                                             const Eigen::Matrix<var,R2,C2>& m2) {
      if (m1.cols() != m2.rows())
        throw std::invalid_argument("m1.cols() != m2.rows()");
      Eigen::Matrix<var,R1,C2> result(m1.rows(),m2.cols());
      for (int i = 0; i < m1.rows(); i++) {
        Eigen::Matrix<var,1,C1> crow(m1.row(i));
        for (int j = 0; j < m2.cols(); j++) {
          Eigen::Matrix<var,R2,1> ccol(m2.col(j));
          result(i,j) = dot_product(crow,ccol);
        }
      }
      return result;
    }

    /**
     * Return the product of the specified matrices.  The number of
     * columns in the first matrix must be the same as the number of rows
     * in the second matrix.
     * @param[in] m1 First matrix.
     * @param[in] m2 Second matrix.
     * @return The product of the first and second matrices.
     * @throw std::invalid_argument if the number of columns of m1 does not match
     *   the number of rows of m2.
     */
    template<int R1,int C1,int R2,int C2>
    inline Eigen::Matrix<var,R1,C2> multiply(const Eigen::Matrix<double,R1,C1>& m1,
                                             const Eigen::Matrix<var,R2,C2>& m2) {
      if (m1.cols() != m2.rows())
        throw std::invalid_argument("m1.cols() != m2.rows()");
      Eigen::Matrix<var,R1,C2> result(m1.rows(),m2.cols());
      for (int i = 0; i < m1.rows(); i++) {
        Eigen::Matrix<double,1,C1> crow(m1.row(i));
        for (int j = 0; j < m2.cols(); j++) {
          Eigen::Matrix<var,R2,1> ccol(m2.col(j));
          result(i,j) = dot_product(crow,ccol);
        }
      }
      return result;
    }
    
    /**
     * Return the product of the specified matrices.  The number of
     * columns in the first matrix must be the same as the number of rows
     * in the second matrix.
     * @param[in] m1 First matrix.
     * @param[in] m2 Second matrix.
     * @return The product of the first and second matrices.
     * @throw std::invalid_argument if the number of columns of m1 does not match
     *   the number of rows of m2.
     */
    template<int R1,int C1,int R2,int C2>
    inline Eigen::Matrix<var,R1,C2> multiply(const Eigen::Matrix<var,R1,C1>& m1,
                                             const Eigen::Matrix<double,R2,C2>& m2) {
      if (m1.cols() != m2.rows())
        throw std::invalid_argument("m1.cols() != m2.rows()");
      Eigen::Matrix<var,R1,C2> result(m1.rows(),m2.cols());
      for (int i = 0; i < m1.rows(); i++) {
        Eigen::Matrix<var,1,C1> crow(m1.row(i));
        for (int j = 0; j < m2.cols(); j++) {
          Eigen::Matrix<double,R2,1> ccol(m2.col(j));
          result(i,j) = dot_product(crow,ccol);
        }
      }
      return result;
    }

    /**
     * Return the scalar product of the specified row vector and
     * specified column vector.  The return is the same as the dot
     * product.  The two vectors must be the same size.
     * @param[in] rv Row vector.
     * @param[in] v Column vector.
     * @return Scalar result of multiplying row vector by column vector.
     * @throw std::invalid_argument if rv and v are not the same size
     */
    template <int C1,int R2>
    inline var multiply(const Eigen::Matrix<var, 1, C1>& rv, 
                        const Eigen::Matrix<var, R2, 1>& v) {
      if (rv.size() != v.size())
        throw std::invalid_argument("rv.size() != v.size()");
      return dot_product(rv, v);
    }
    /**
     * Return the scalar product of the specified row vector and
     * specified column vector.  The return is the same as the dot
     * product.  The two vectors must be the same size.
     * @param[in] rv Row vector.
     * @param[in] v Column vector.
     * @return Scalar result of multiplying row vector by column vector.
     * @throw std::invalid_argument if rv and v are not the same size
     */
    template <int C1,int R2>
    inline var multiply(const Eigen::Matrix<double, 1, C1>& rv, 
                        const Eigen::Matrix<var, R2, 1>& v) {
      if (rv.size() != v.size())
        throw std::invalid_argument("rv.size() != v.size()");
      return dot_product(rv, v);
    }
    /**
     * Return the scalar product of the specified row vector and
     * specified column vector.  The return is the same as the dot
     * product.  The two vectors must be the same size.
     * @param[in] rv Row vector.
     * @param[in] v Column vector.
     * @return Scalar result of multiplying row vector by column vector.
     * @throw std::invalid_argument if rv and v are not the same size
     */
    template <int C1,int R2>
    inline var multiply(const Eigen::Matrix<var, 1, C1>& rv, 
                        const Eigen::Matrix<double, R2, 1>& v) {
      if (rv.size() != v.size())
        throw std::invalid_argument("rv.size() != v.size()");
      return dot_product(rv, v);
    }

    /**
     * Return the specified row minus 1 of the specified matrix.  That is,
     * indexing is from 1, not 0, so this function returns the same
     * value as the Eigen matrix call <code>m.row(i - 1)</code>.
     * @param[in] m Matrix.
     * @param[in] i Row index (plus 1).
     * @return Specified row of the matrix; between 1 and the number
     * of rows of <code>m</code> inclusive.
     * @throws std::invalid_argument If the index is 0 or
     * greater than the number of columns.
     */
    row_vector_v row(const matrix_v& m, size_t i);

    /**
     * Return the specified column minus 1 of the specified matrix.  Thus
     * indexing is from 1, not 0, and this method returns the equivalent of
     * the Eigen matrix call <code>m.col(j - 1)</code>.
     * @param[in] m Matrix.
     * @param[in] j Column index (plus 1); between 1 and the number of
     * columns of <code>m</code> inclusive.
     * @return Specified column of the matrix.
     * @throws std::invalid_argument if the index is 0 or greater than
     * the number of columns.
     */
    vector_v col(const matrix_v& m, size_t j);

    /**
     * Return a column vector of the diagonal elements of the
     * specified matrix.  The matrix is not required to be square.
     * @param[in] m Specified matrix.  
     * @return Diagonal of the matrix.
     */
    vector_v diagonal(const matrix_v& m);

    /**
     * Return a square diagonal matrix with the specified vector of
     * coefficients as the diagonal values.
     * @param[in] v Specified vector.
     * @return Diagonal matrix with vector as diagonal values.
     */
    matrix_v diag_matrix(const vector_v& v);

    /**
     * Return the transposition of the specified column
     * vector.
     * @param[in] v Specified vector.
     * @return Transpose of the vector.
     */
    row_vector_v transpose(const vector_v& v);
    /**
     * Return the transposition of the specified row
     * vector.
     * @param[in] rv Specified vector.
     * @return Transpose of the vector.
     */
    vector_v transpose(const row_vector_v& rv);
    /**
     * Return the transposition of the specified matrix.
     * @param[in] m Specified matrix.
     * @return Transpose of the matrix.
     */
    matrix_v transpose(const matrix_v& m);

    /**
     * Returns the inverse of the specified matrix.
     * @param[in] m Specified matrix.
     * @return Inverse of the matrix.
     */
    matrix_v inverse(const matrix_v& m);

    /**
     * Return the softmax of the specified vector.
     * @param x Vector to transform
     * @return Unit simplex result of the softmax transform of the vector.
     */
    vector_v softmax(const vector_v& v);

    /**
     * Returns the solution of the system Ax=b when A is triangular.
     * @param[in] A Triangular matrix.  Upper or lower is defined by TriView being
     * either Eigen::Upper or Eigen::Lower.
     * @param[in] b Right hand side matrix or vector.
     * @return x = A^-1 b, solution of the linear system.
     * @throws std::invalid_argument if A is not square or the rows of b don't
     * match the size of A.
     */
    template<int TriView,int R1,int C1,int R2,int C2>
    inline Eigen::Matrix<var,R1,C2> mdivide_left_tri(const Eigen::Matrix<var,R1,C1> &A,
                                                     const Eigen::Matrix<var,R2,C2> &b) {
      if (A.cols() != A.rows())
        throw std::invalid_argument("A is not square");
      if (A.cols() != b.rows())
        throw std::invalid_argument("A.cols() != b.rows()");
      return A.template triangularView<TriView>().solve(b);
    }
    /**
     * Returns the solution of the system Ax=b when A is triangular.
     * @param[in] A Triangular matrix.  Upper or lower is defined by TriView being
     * either Eigen::Upper or Eigen::Lower.
     * @param[in] b Right hand side matrix or vector.
     * @return x = A^-1 b, solution of the linear system.
     * @throws std::invalid_argument if A is not square or the rows of b don't
     * match the size of A.
     */
    template<int TriView,int R1,int C1,int R2,int C2>
    inline Eigen::Matrix<var,R1,C2> mdivide_left_tri(const Eigen::Matrix<var,R1,C1> &A,
                                                     const Eigen::Matrix<double,R2,C2> &b) {
      if (A.cols() != A.rows())
        throw std::invalid_argument("A is not square");
      if (A.cols() != b.rows())
        throw std::invalid_argument("A.cols() != b.rows()");
      return A.template triangularView<TriView>().solve(to_var(b));
    }
    /**
     * Returns the solution of the system Ax=b when A is triangular.
     * @param[in] A Triangular matrix.  Upper or lower is defined by TriView being
     * either Eigen::Upper or Eigen::Lower.
     * @param[in] b Right hand side matrix or vector.
     * @return x = A^-1 b, solution of the linear system.
     * @throws std::invalid_argument if A is not square or the rows of b don't
     * match the size of A.
     */
    template<int TriView,int R1,int C1,int R2,int C2>
    inline Eigen::Matrix<var,R1,C2> mdivide_left_tri(const Eigen::Matrix<double,R1,C1> &A,
                                                     const Eigen::Matrix<var,R2,C2> &b) {
      if (A.cols() != A.rows())
        throw std::invalid_argument("A is not square");
      if (A.cols() != b.rows())
        throw std::invalid_argument("A.cols() != b.rows()");
      return to_var(A).template triangularView<TriView>().solve(b);
    }

    /**
     * Returns the solution of the system Ax=b.
     * @param[in] A Matrix.
     * @param[in] b Right hand side matrix or vector.
     * @return x = A^-1 b, solution of the linear system.
     * @throws std::invalid_argument if A is not square or the rows of b don't
     * match the size of A.
     */
    template<int R1,int C1,int R2,int C2>
    inline Eigen::Matrix<var,R1,C2> mdivide_left(const Eigen::Matrix<var,R1,C1> &A,
                                                 const Eigen::Matrix<var,R2,C2> &b) {
      if (A.cols() != A.rows())
        throw std::invalid_argument("A is not square");
      if (A.cols() != b.rows())
        throw std::invalid_argument("A.cols() != b.rows()");
      return A.lu().solve(b);
    }
    /**
     * Returns the solution of the system Ax=b.
     * @param[in] A Matrix.
     * @param[in] b Right hand side matrix or vector.
     * @return x = A^-1 b, solution of the linear system.
     * @throws std::invalid_argument if A is not square or the rows of b don't
     * match the size of A.
     */
    template<int R1, int C1, int R2, int C2>
    inline Eigen::Matrix<var,R1,C2> mdivide_left(const Eigen::Matrix<double,R1,C1> &A,
                                                 const Eigen::Matrix<var,R2,C2> &b) {
      if (A.cols() != A.rows())
        throw std::invalid_argument("A is not square");
      if (A.cols() != b.rows())
        throw std::invalid_argument("A.cols() != b.rows()");
      // FIXME: it would be much faster to do LU, then convert to var
      return to_var(A).lu().solve(b);
    }
    /**
     * Returns the solution of the system Ax=b.
     * @param[in] A Matrix.
     * @param[in] b Right hand side matrix or vector.
     * @return x = A^-1 b, solution of the linear system.
     * @throws std::invalid_argument if A is not square or the rows of b don't
     * match the size of A.
     */
    template<int R1, int C1, int R2, int C2>
    inline Eigen::Matrix<var,R1,C2> mdivide_left(const Eigen::Matrix<var,R1,C1> &A,
                                                 const Eigen::Matrix<double,R2,C2> &b) {
      if (A.cols() != A.rows())
        throw std::invalid_argument("A is not square");
      if (A.cols() != b.rows())
        throw std::invalid_argument("A.cols() != b.rows()");
      return A.lu().solve(to_var(b));
    }

    /**
     * Returns the solution x of the system xA = b.
     * @param[in] b Right hand side matrix or vector.
     * @param[in] A Matrix.
     * @return x = b A^-1, solution of the linear system.
     * @throws std::invalid_argument if A is not square or the cols of b don't
     * match the size of A.
     */
    template<int R1,int C1,int R2,int C2>
    inline Eigen::Matrix<var,R1,C2> mdivide_right(const Eigen::Matrix<var,R1,C1> &b,
                                                  const Eigen::Matrix<var,R2,C2> &A) {
      if (A.cols() != A.rows())
        throw std::invalid_argument("A is not square");
      if (A.rows() != b.cols())
        throw std::invalid_argument("A.rows() != b.cols()");
      return A.transpose().lu().solve(b.transpose()).transpose();
    }
    /**
     * Returns the solution x of the system xA = b.
     * @param[in] b Right hand side matrix or vector.
     * @param[in] A Matrix.
     * @return x = b A^-1, solution of the linear system.
     * @throws std::invalid_argument if A is not square or the cols of b don't
     * match the size of A.
     */
    template<int R1,int C1,int R2,int C2>
    inline Eigen::Matrix<var,R1,C2> mdivide_right(const Eigen::Matrix<double,R1,C1> &b,
                                                  const Eigen::Matrix<var,R2,C2> &A) {
      if (A.cols() != A.rows())
        throw std::invalid_argument("A is not square");
      if (A.rows() != b.cols())
        throw std::invalid_argument("A.rows() != b.cols()");
      return A.transpose().lu().solve(to_var(b).transpose()).transpose();
    }
    /**
     * Returns the solution x of the system xA = b.
     * @param[in] b Right hand side matrix or vector.
     * @param[in] A Matrix.
     * @return x = b A^-1, solution of the linear system.
     * @throws std::invalid_argument if A is not square or the cols of b don't
     * match the size of A.
     */
    template<int R1,int C1,int R2,int C2>
    inline Eigen::Matrix<var,R1,C2> mdivide_right(const Eigen::Matrix<var,R1,C1> &b,
                                                  const Eigen::Matrix<double,R2,C2> &A) {
      if (A.cols() != A.rows())
        throw std::invalid_argument("A is not square");
      if (A.rows() != b.cols())
        throw std::invalid_argument("A.rows() != b.cols()");
      return to_var(A).transpose().lu().solve(b.transpose()).transpose();
    }
    /**
     * Return the real component of the eigenvalues of the specified
     * matrix in descending order of magnitude.
     * <p>See <code>eigen_decompose()</code> for more information.
     * @param[in] m Specified matrix.
     * @return Eigenvalues of matrix.
     */
    vector_v eigenvalues(const matrix_v& m);

    /**
     * Return a matrix whose columns are the real components of the
     * eigenvectors of the specified matrix.
     * <p>See <code>eigen_decompose()</code> for more information.
     * @param[in] m Specified matrix.
     * @return Eigenvectors of matrix.
     */
    matrix_v eigenvectors(const matrix_v& m);
    /**
     * Assign the real components of the eigenvalues and eigenvectors
     * of the specified matrix to the specified references.
     * <p>Given an input matrix \f$A\f$, the
     * eigenvalues will be found in \f$D\f$ in descending order of
     * magnitude.  The eigenvectors will be written into
     * the columns of \f$V\f$.  If \f$A\f$ is invertible, then
     * <p>\f$A = V \times \mbox{\rm diag}(D) \times V^{-1}\f$, where
     $ \f$\mbox{\rm diag}(D)\f$ is the square diagonal matrix with
     * diagonal elements \f$D\f$ and \f$V^{-1}\f$ is the inverse of
     * \f$V\f$.
     * @param[in] m Specified matrix.
     * @param[out] eigenvalues Column vector reference into which
     * eigenvalues are written.
     * @param[out] eigenvectors Matrix reference into which eigenvectors
     * are written.
     */
    void eigen_decompose(const matrix_v& m,
                         vector_v& eigenvalues,
                         matrix_v& eigenvectors);

    /**
     * Return the eigenvalues of the specified symmetric matrix
     * in descending order of magnitude.  This function is more
     * efficient than the general eigenvalues function for symmetric
     * matrices.
     * <p>See <code>eigen_decompose()</code> for more information.
     * @param[in] m Specified matrix.
     * @return Eigenvalues of matrix.
     */
    vector_v eigenvalues_sym(const matrix_v& m);
    /**
     * Return a matrix whose rows are the real components of the
     * eigenvectors of the specified symmetric matrix.  This function
     * is more efficient than the general eigenvectors function for
     * symmetric matrices.
     * <p>See <code>eigen_decompose()</code> for more information.
     * @param[in] m Symmetric matrix.
     * @return Eigenvectors of matrix.
     */
    matrix_v eigenvectors_sym(const matrix_v& m);
    /**
     * Assign the real components of the eigenvalues and eigenvectors
     * of the specified symmetric matrix to the specified references.
     * <p>See <code>eigen_decompose()</code> for more information on the
     * values.
     * @param[in] m Symmetric matrix.  This function is more efficient
     * than the general decomposition method for symmetric matrices.
     * @param[out] eigenvalues Column vector reference into which
     * eigenvalues are written.
     * @param[out] eigenvectors Matrix reference into which eigenvectors
     * are written.
     */
    void eigen_decompose_sym(const matrix_v& m,
                             vector_v& eigenvalues,
                             matrix_v& eigenvectors);

    /**
     * Return the lower-triangular Cholesky factor (i.e., matrix
     * square root) of the specified square, symmetric matrix.  The return
     * value \f$L\f$ will be a lower-traingular matrix such that the
     * original matrix \f$A\f$ is given by
     * <p>\f$A = L \times L^T\f$.
     * @param[in] m Symmetrix matrix.
     * @return Square root of matrix.
     * @throw std::domain_error if m is not a square matrix
     */
    matrix_v cholesky_decompose(const matrix_v& m);

    /**
     * Return the vector of the singular values of the specified matrix
     * in decreasing order of magnitude.
     * <p>See the documentation for <code>svd()</code> for
     * information on the signular values.
     * @param[in] m Specified matrix.
     * @return Singular values of the matrix.
     */
    vector_v singular_values(const matrix_v& m);

    /**
     * Assign the real components of a singular value decomposition
     * of the specified matrix to the specified references.  
     * <p>Thesingular values \f$S\f$ are assigned to a vector in 
     * decreasing order of magnitude.  The left singular vectors are
     * found in the columns of \f$U\f$ and the right singular vectors
     * in the columns of \f$V\f$.
     * <p>The original matrix is recoverable as
     * <p>\f$A = U \times \mbox{\rm diag}(S) \times V^T\f$, where
     * \f$\mbox{\rm diag}(S)\f$ is the square diagonal matrix with
     * diagonal elements \f$S\f$.
     * <p>If \f$A\f$ is an \f$M \times N\f$ matrix
     * and \f$K = \mbox{\rm min}(M,N)\f$, 
     * then \f$U\f$ is an \f$M \times K\f$ matrix,  
     * \f$S\f$ is a length \f$K\f$ column vector, and 
     * \f$V\f$ is an \f$N \times K\f$ matrix.
     * @param[in] m Matrix to decompose.
     * @param[out] u Left singular vectors.
     * @param[out] v Right singular vectors.
     * @param[out] s Singular values.
     */
    void svd(const matrix_v& m,
             matrix_v& u,
             matrix_v& v,
             vector_v& s);
  }
}


namespace Eigen {

  namespace internal {

    // FIXME: untested
    /**
     * Template specification of general_matrix_vector_product for stan::agrad::var.
     */
    template<typename Index, bool ConjugateLhs, bool ConjugateRhs>
    struct general_matrix_vector_product<Index,stan::agrad::var,ColMajor,ConjugateLhs,stan::agrad::var,ConjugateRhs>
    {
      typedef stan::agrad::var LhsScalar;
      typedef stan::agrad::var RhsScalar;
      typedef typename scalar_product_traits<LhsScalar, RhsScalar>::ReturnType ResScalar;
      enum { LhsStorageOrder = ColMajor };

      EIGEN_DONT_INLINE static void run(
                                        Index rows, Index cols,
                                        const LhsScalar* lhs, Index lhsStride,
                                        const RhsScalar* rhs, Index rhsIncr,
                                        ResScalar* res, Index resIncr, const ResScalar &alpha)
      {
        for (Index i = 0; i < rows; i++) {
          res[i*resIncr] += stan::agrad::var(new stan::agrad::gevv_vvv_vari(&alpha,(int(LhsStorageOrder) == int(ColMajor))?(&lhs[i]):(&lhs[i*lhsStride]),(int(LhsStorageOrder) == int(ColMajor))?(lhsStride):(1),rhs,rhsIncr,cols));
        }
      }
    };
    
    // FIXME: untested
    /**
     * Template specification of general_matrix_vector_product for stan::agrad::var.
     */
    template<typename Index, bool ConjugateLhs, bool ConjugateRhs>
    struct general_matrix_vector_product<Index,stan::agrad::var,RowMajor,ConjugateLhs,stan::agrad::var,ConjugateRhs>
    {
      typedef stan::agrad::var LhsScalar;
      typedef stan::agrad::var RhsScalar;
      typedef typename scalar_product_traits<LhsScalar, RhsScalar>::ReturnType ResScalar;
      enum { LhsStorageOrder = RowMajor };

      EIGEN_DONT_INLINE static void run(
                                        Index rows, Index cols,
                                        const LhsScalar* lhs, Index lhsStride,
                                        const RhsScalar* rhs, Index rhsIncr,
                                        ResScalar* res, Index resIncr, const RhsScalar &alpha)
      {
        for (Index i = 0; i < rows; i++) {
          res[i*resIncr] += stan::agrad::var(new stan::agrad::gevv_vvv_vari(&alpha,(int(LhsStorageOrder) == int(ColMajor))?(&lhs[i]):(&lhs[i*lhsStride]),(int(LhsStorageOrder) == int(ColMajor))?(lhsStride):(1),rhs,rhsIncr,cols));
        }
      }
    };
  }
}


#endif

