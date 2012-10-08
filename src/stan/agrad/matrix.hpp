#ifndef __STAN__AGRAD__MATRIX_HPP__
#define __STAN__AGRAD__MATRIX_HPP__

// global include
#include <sstream>

#include <stan/agrad/agrad.hpp>
#include <stan/agrad/special_functions.hpp>
#include <stan/math/boost_error_handling.hpp>
#include <stan/math/matrix.hpp>

namespace stan {
  namespace agrad {
    class gevv_vvv_vari : public stan::agrad::vari {
    protected:
      stan::agrad::vari* alpha_;
      stan::agrad::vari** v1_;
      stan::agrad::vari** v2_;
      double dotval_;
      size_t length_;
      inline static double eval_gevv(const stan::agrad::var* alpha,
                                     const stan::agrad::var* v1, int stride1,
                                     const stan::agrad::var* v2, int stride2,
                                     size_t length, double *dotprod) {
        double result = 0;
        for (size_t i = 0; i < length; i++)
          result += v1[i*stride1].vi_->val_ * v2[i*stride2].vi_->val_;
        *dotprod = result;
        return alpha->vi_->val_ * result;
      }
    public:
      gevv_vvv_vari(const stan::agrad::var* alpha, 
                    const stan::agrad::var* v1, int stride1, 
                    const stan::agrad::var* v2, int stride2, size_t length) : 
        vari(eval_gevv(alpha,v1,stride1,v2,stride2,length,&dotval_)), length_(length) {
        alpha_ = alpha->vi_;
        v1_ = (stan::agrad::vari**)stan::agrad::memalloc_.alloc(2*length_*sizeof(stan::agrad::vari*));
        v2_ = v1_ + length_;
        for (size_t i = 0; i < length_; i++)
          v1_[i] = v1[i*stride1].vi_;
        for (size_t i = 0; i < length_; i++)
          v2_[i] = v2[i*stride2].vi_;
      }
      void chain() {
        const double adj_alpha = adj_ * alpha_->val_;
        for (size_t i = 0; i < length_; i++) {
          v1_[i]->adj_ += adj_alpha * v2_[i]->val_;
          v2_[i]->adj_ += adj_alpha * v1_[i]->val_;
        }
        alpha_->adj_ += adj_ * dotval_;
      }
    };
  }
}

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

    /**
     * Override matrix-vector and matrix-matrix products to use more efficient implementation.
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
          res[i*resIncr] += stan::agrad::var(new stan::agrad::gevv_vvv_vari(&alpha,((int)LhsStorageOrder == (int)ColMajor)?(&lhs[i]):(&lhs[i*lhsStride]),((int)LhsStorageOrder == (int)ColMajor)?(lhsStride):(1),rhs,rhsIncr,cols));
        }
      }
    };
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
          res[i*resIncr] += stan::agrad::var(new stan::agrad::gevv_vvv_vari(&alpha,((int)LhsStorageOrder == (int)ColMajor)?(&lhs[i]):(&lhs[i*lhsStride]),((int)LhsStorageOrder == (int)ColMajor)?(lhsStride):(1),rhs,rhsIncr,cols));
        }
      }
    };
    template<typename Index, int LhsStorageOrder, bool ConjugateLhs, int RhsStorageOrder, bool ConjugateRhs>
    struct general_matrix_matrix_product<Index,stan::agrad::var,LhsStorageOrder,ConjugateLhs,stan::agrad::var,RhsStorageOrder,ConjugateRhs,ColMajor>
    {
      typedef stan::agrad::var LhsScalar;
      typedef stan::agrad::var RhsScalar;
      typedef typename scalar_product_traits<LhsScalar, RhsScalar>::ReturnType ResScalar;
      static void run(Index rows, Index cols, Index depth,
                      const LhsScalar* _lhs, Index lhsStride,
                      const RhsScalar* _rhs, Index rhsStride,
                      ResScalar* res, Index resStride,
                      const ResScalar &alpha,
                      level3_blocking<LhsScalar,RhsScalar>& blocking,
                      GemmParallelInfo<Index>* info = 0)
      {
        for (Index i = 0; i < cols; i++) {
          general_matrix_vector_product<Index,LhsScalar,LhsStorageOrder,ConjugateLhs,RhsScalar,ConjugateRhs>::run(
              rows,depth,_lhs,lhsStride,
              &_rhs[((int)RhsStorageOrder == (int)ColMajor)?(i*rhsStride):(i)],((int)RhsStorageOrder == (int)ColMajor)?(1):(rhsStride),
              &res[i*resStride],1,alpha);
        }
      }
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
     * Initialize variable to value.  (Function may look pointless, but
     * its needed to bottom out recursion.)
     */
    inline void initialize_variable(var& variable, const var& value) {
      variable = value;
    }

    /**
     * Initialize every cell in the matrix to the specified value.
     * 
     */
    template <int R, int C>
    inline void initialize_variable(Eigen::Matrix<var,R,C>& matrix, const var& value) {
      for (int i = 0; i < matrix.size(); ++i)
        matrix(i) = value;
    }

    /**
     * Initialize the variables in the standard vector recursively.
     */
    template <typename T>
    inline void initialize_variable(std::vector<T>& variables, const var& value) {
      for (size_t i = 0; i < variables.size(); ++i)
        initialize_variable(variables[i],value);
    }


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
        dot_self_vari(vari** v, size_t size) 
          : vari(var_dot_self(v,size)), 
            v_(v),
            size_(size) {
        }
        template <int R, int C>
        dot_self_vari(const Eigen::Matrix<var,R,C>& v) :
          vari(var_dot_self(v)), size_(v.size()) {
          v_ = (vari**) memalloc_.alloc(size_ * sizeof(vari*));
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
          v_ = (vari**)memalloc_.alloc(length_*sizeof(vari*));
          for (size_t i = 0; i < length_; i++)
            v_[i] = v(i).vi_;
        }
        template<int R1,int C1>
        sum_v_vari(const Eigen::Matrix<var,R1,C1> &v1) :
          vari(var_sum(v1)), length_(v1.size()) {
          v_ = (vari**)memalloc_.alloc(length_*sizeof(vari*));
          for (size_t i = 0; i < length_; i++)
            v_[i] = v1(i).vi_;
        }
        sum_v_vari(const var *v, size_t len) :
          vari(var_sum(v,len)), length_(len) {
          v_ = (vari**)memalloc_.alloc(length_*sizeof(vari*));
          for (size_t i = 0; i < length_; i++)
            v_[i] = v[i].vi_;
        }
        void chain() {
          for (size_t i = 0; i < length_; i++) {
            v_[i]->adj_ += adj_;
          }
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
        void chain() {
          for (size_t i = 0; i < length_; i++) {
            v1_[i]->adj_ += adj_ * v2_[i];
          }
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
      stan::math::validate_vector(v,"dot_self");
      return var(new dot_self_vari(v));
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

    /**
     * Returns the sum of the coefficients of the specified
     * column vector.
     * @param v Specified vector.
     * @return Sum of coefficients of vector.
     */
    template <int R, int C>
    inline var sum(const Eigen::Matrix<var,R,C>& m) {
      if (m.size() == 0)
        return 0.0;
      return var(new sum_v_vari(m));
    }


     
    /**
     * Return the sum of the specified column vectors.
     * The two vectors must have the same size.
     *
     * @param[in] v1 First vector.
     * @param[in] v2 Second vector.
     * @return Sum of the two vectors.
     * @throw std::domain_error if size of v1 is not equal to size of v2.
     */
    template <typename T1, typename T2, int R, int C>
    inline Eigen::Matrix<typename boost::math::tools::promote_args<T1,T2>::type, R, C>
    add(const Eigen::Matrix<T1, R, C>& v1, 
        const Eigen::Matrix<T2, R, C>& v2) {
      stan::math::validate_matching_dims(v1,v2,"add");
      return to_var(v1) + to_var(v2);
    }
    /**
     * Return the sum of a matrix or vector and a scalar.
     * @param[in] m Matrix or vector.
     * @param[in] c Scalar.
     * @return Matrix or Vector plus the scalar.
     */
    template <typename T1, typename T2, int R, int C>
    inline Eigen::Matrix<typename boost::math::tools::promote_args<T1,T2>::type, R, C>
    add(const Eigen::Matrix<T1, R, C>& m,
        const T2& c) {
      return (to_var(m).array() + to_var(c)).matrix();
    }
    /**
     * Return the sum of a scalar and a matrix or vector.
     * @param[in] c Scalar.
     * @param[in] m Matrix or vector.
     * @return Scalar plus vector.
     */
    template <typename T1, typename T2, int R, int C>
    inline Eigen::Matrix<typename boost::math::tools::promote_args<T1,T2>::type, R, C>
    add(const T1& c,
        const Eigen::Matrix<T2, R, C>& m) {
      return (to_var(c) + to_var(m).array()).matrix();
    }


    /**
     * Return the difference between the first specified column vector
     * and the second.  The two vectors must have the same size.
     * @param[in] v1 First vector.
     * @param[in] v2 Second vector.
     * @return First vector minus the second vector.
     * @throw std::domain_error if size of v1 does not match size of v2.
     */
    template <typename T1, typename T2, int R, int C>
    inline Eigen::Matrix<typename boost::math::tools::promote_args<T1,T2>::type, R, C>
    subtract(const Eigen::Matrix<T1, R, C>& v1, 
             const Eigen::Matrix<T2, R, C>& v2) {
      stan::math::validate_matching_dims(v1,v2,"subtract");
      return to_var(v1) - to_var(v2);
    }
    /**
     * Return the difference between a matrix or vector  and a scalar.
     * @param[in] m Matrix or vector.
     * @param[in] c Scalar.
     * @return Vector minus the scalar.
     */
    template <typename T1, typename T2, int R, int C>
    inline Eigen::Matrix<typename boost::math::tools::promote_args<T1,T2>::type, R, C>
    subtract(const Eigen::Matrix<T1, R, C>& m,
             const T2& c) {
      return (to_var(m).array() - to_var(c)).matrix();
    }
    /**
     * Return the difference between a scalar and a matrix or vector.
     * @param[in] c Scalar.
     * @param[in] m Matrix or vector.
     * @return Scalar minus vector.
     */
    template <typename T1, typename T2, int R, int C>
    inline Eigen::Matrix<typename boost::math::tools::promote_args<T1,T2>::type, R, C>
    subtract(const T1& c,
             const Eigen::Matrix<T2, R, C>& m) {
      return (to_var(c) - to_var(m).array()).matrix();
    }

    /**
     * Return the negation of the specified variable.  The result
     * is the same as multiplying by the scalar <code>-1</code>.
     * @param[in] v Specified variable.  
     * @return The negation of the variable.
     */
    template <typename T>
    inline T minus(const T& v) {
      return -v;
    }
    /**
     * Return the negation of the specified column vector.  The result
     * is the same as multiplying by the scalar <code>-1</code>.
     * @param[in] v Specified vector.  
     * @return The negation of the vector.
     */
    template <typename T, int R, int C>
    inline Eigen::Matrix<T,R,C> minus(const Eigen::Matrix<T, R, C>& v) {
      return -v;
    }

    /**
     * Return the division of the first scalar by
     * the second scalar.
     * @param[in] v Specified vector.
     * @param[in] c Specified scalar.
     * @return Vector divided by the scalar.
     */
    inline double
    divide(double x, double y) { 
      return x / y; 
    }
    template <typename T1, typename T2>
    inline var
    divide(const T1& v, const T2& c) {
      return to_var(v) / to_var(c);
    }
    /**
     * Return the division of the specified column vector by
     * the specified scalar.
     * @param[in] v Specified vector.
     * @param[in] c Specified scalar.
     * @return Vector divided by the scalar.
     */
    template <typename T1, typename T2, int R, int C>
    inline Eigen::Matrix<var,R,C>
    divide(const Eigen::Matrix<T1, R,C>& v, const T2& c) {
      return to_var(v) / to_var(c);
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
      stan::math::validate_matching_dims(m1,m2,"elt_multiply");
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
      stan::math::validate_matching_dims(v1,v2,"elt_multiply");
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
      stan::math::validate_matching_dims(v1,v2,"elt_multiply");
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
      stan::math::validate_matching_dims(m1,m2,"elt_divide");
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
      stan::math::validate_matching_dims(v1,v2,"elt_divide");
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
      stan::math::validate_matching_dims(v1,v2,"elt_divide");
      Eigen::Matrix<stan::agrad::var,1,Eigen::Dynamic> result(v1.size());
      for (int i = 0; i < v1.size(); ++i)
        result(i) = v1(i) / v2(i);
      return result;
    }
    
    /**
     * Return the product of two scalars.
     * @param[in] v First scalar.
     * @param[in] c Specified scalar.
     * @return Product of scalars.
     */
    template <typename T1, typename T2>
    inline
    typename boost::math::tools::promote_args<T1,T2>::type
    multiply(const T1& v, const T2& c) {
      return v * c;
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
      // FIXME:  pull out to eliminate overpromotion of one side
      // move to matrix.hpp w. promotion?
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
     * @throw std::domain_error if the number of columns of m1 does not match
     *   the number of rows of m2.
     */
    template<int R1,int C1,int R2,int C2>
    inline Eigen::Matrix<var,R1,C2> multiply(const Eigen::Matrix<var,R1,C1>& m1,
                                             const Eigen::Matrix<var,R2,C2>& m2) {
      stan::math::validate_multiplicable(m1,m2,"multiply");
      Eigen::Matrix<var,R1,C2> result(m1.rows(),m2.cols());
      for (int i = 0; i < m1.rows(); i++) {
        typename Eigen::Matrix<var,R1,C1>::ConstRowXpr crow(m1.row(i));
        for (int j = 0; j < m2.cols(); j++) {
          typename Eigen::Matrix<var,R2,C2>::ConstColXpr ccol(m2.col(j));
          if (j == 0) {
            if (i == 0) {
              result(i,j) = var(new dot_product_vv_vari(crow,ccol));
            }
            else {
              dot_product_vv_vari *v2 = static_cast<dot_product_vv_vari*>(result(0,j).vi_);
              result(i,j) = var(new dot_product_vv_vari(crow,ccol,NULL,v2));
            }
          }
          else { 
            if (i == 0) {
              dot_product_vv_vari *v1 = static_cast<dot_product_vv_vari*>(result(i,0).vi_);
              result(i,j) = var(new dot_product_vv_vari(crow,ccol,v1));
            }
            else /* if (i != 0 && j != 0) */ {
              dot_product_vv_vari *v1 = static_cast<dot_product_vv_vari*>(result(i,0).vi_);
              dot_product_vv_vari *v2 = static_cast<dot_product_vv_vari*>(result(0,j).vi_);
              result(i,j) = var(new dot_product_vv_vari(crow,ccol,v1,v2));
            }
          }
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
     * @throw std::domain_error if the number of columns of m1 does not match
     *   the number of rows of m2.
     */
    template<int R1,int C1,int R2,int C2>
    inline Eigen::Matrix<var,R1,C2> multiply(const Eigen::Matrix<double,R1,C1>& m1,
                                             const Eigen::Matrix<var,R2,C2>& m2) {
      stan::math::validate_multiplicable(m1,m2,"multiply");
      Eigen::Matrix<var,R1,C2> result(m1.rows(),m2.cols());
      for (int i = 0; i < m1.rows(); i++) {
        typename Eigen::Matrix<double,R1,C1>::ConstRowXpr crow(m1.row(i));
        for (int j = 0; j < m2.cols(); j++) {
          typename Eigen::Matrix<var,R2,C2>::ConstColXpr ccol(m2.col(j));
//          result(i,j) = dot_product(crow,ccol);
          if (j == 0) {
            if (i == 0) {
              result(i,j) = var(new dot_product_vd_vari(ccol,crow));
            }
            else {
              dot_product_vd_vari *v2 = static_cast<dot_product_vd_vari*>(result(0,j).vi_);
              result(i,j) = var(new dot_product_vd_vari(ccol,crow,v2,NULL));
            }
          }
          else { 
            if (i == 0) {
              dot_product_vd_vari *v1 = static_cast<dot_product_vd_vari*>(result(i,0).vi_);
              result(i,j) = var(new dot_product_vd_vari(ccol,crow,NULL,v1));
            }
            else /* if (i != 0 && j != 0) */ {
              dot_product_vd_vari *v1 = static_cast<dot_product_vd_vari*>(result(i,0).vi_);
              dot_product_vd_vari *v2 = static_cast<dot_product_vd_vari*>(result(0,j).vi_);
              result(i,j) = var(new dot_product_vd_vari(ccol,crow,v2,v1));
            }
          }
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
     * @throw std::domain_error if the number of columns of m1 does not match
     *   the number of rows of m2.
     */
    template<int R1,int C1,int R2,int C2>
    inline Eigen::Matrix<var,R1,C2> multiply(const Eigen::Matrix<var,R1,C1>& m1,
                                             const Eigen::Matrix<double,R2,C2>& m2) {
      stan::math::validate_multiplicable(m1,m2,"multiply");
      Eigen::Matrix<var,R1,C2> result(m1.rows(),m2.cols());
      for (int i = 0; i < m1.rows(); i++) {
        typename Eigen::Matrix<var,R1,C1>::ConstRowXpr crow(m1.row(i));
        for (int j = 0; j < m2.cols(); j++) {
          typename Eigen::Matrix<double,R2,C2>::ConstColXpr ccol(m2.col(j));
//          result(i,j) = dot_product(crow,ccol);
          if (j == 0) {
            if (i == 0) {
              result(i,j) = var(new dot_product_vd_vari(crow,ccol));
            }
            else {
              dot_product_vd_vari *v2 = static_cast<dot_product_vd_vari*>(result(0,j).vi_);
              result(i,j) = var(new dot_product_vd_vari(crow,ccol,NULL,v2));
            }
          }
          else { 
            if (i == 0) {
              dot_product_vd_vari *v1 = static_cast<dot_product_vd_vari*>(result(i,0).vi_);
              result(i,j) = var(new dot_product_vd_vari(crow,ccol,v1,NULL));
            }
            else /* if (i != 0 && j != 0) */ {
              dot_product_vd_vari *v1 = static_cast<dot_product_vd_vari*>(result(i,0).vi_);
              dot_product_vd_vari *v2 = static_cast<dot_product_vd_vari*>(result(0,j).vi_);
              result(i,j) = var(new dot_product_vd_vari(crow,ccol,v1,v2));
            }
          }
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
     * @throw std::domain_error if rv and v are not the same size
     */
    template <int C1,int R2>
    inline var multiply(const Eigen::Matrix<var, 1, C1>& rv, 
                        const Eigen::Matrix<var, R2, 1>& v) {
      if (rv.size() != v.size())
        throw std::domain_error("row vector and vector must be same length in multiply");
      return dot_product(rv, v);
    }
    /**
     * Return the scalar product of the specified row vector and
     * specified column vector.  The return is the same as the dot
     * product.  The two vectors must be the same size.
     * @param[in] rv Row vector.
     * @param[in] v Column vector.
     * @return Scalar result of multiplying row vector by column vector.
     * @throw std::domain_error if rv and v are not the same size
     */
    template <int C1,int R2>
    inline var multiply(const Eigen::Matrix<double, 1, C1>& rv, 
                        const Eigen::Matrix<var, R2, 1>& v) {
      stan::math::validate_multiplicable(rv,v,"multiply");
      return dot_product(rv, v);
    }
    /**
     * Return the scalar product of the specified row vector and
     * specified column vector.  The return is the same as the dot
     * product.  The two vectors must be the same size.
     * @param[in] rv Row vector.
     * @param[in] v Column vector.
     * @return Scalar result of multiplying row vector by column vector.
     * @throw std::domain_error if rv and v are not the same size
     */
    template <int C1,int R2>
    inline var multiply(const Eigen::Matrix<var, 1, C1>& rv, 
                        const Eigen::Matrix<double, R2, 1>& v) {
      stan::math::validate_multiplicable(rv,v,"multiply");
      return dot_product(rv, v);
    }

    inline matrix_v 
    multiply_lower_tri_self_transpose(const matrix_v& L) {
      stan::math::validate_square(L,"multiply_lower_tri_self_transpose");
      int K = L.rows();
      matrix_v LLt(K,K);
      if (K == 0) return LLt;
      // if (K == 1) {
      //   LLt(0,0) = L(0,0) * L(0,0);
      //   return LLt;
      // }
      int Knz = (K * (K + 1)) / 2;  // nonzero: (K choose 2) below
                                    // diag + K on diag
      vari** vs = (vari**)memalloc_.alloc( Knz * sizeof(vari*) );
      int pos = 0;
      for (int m = 0; m < K; ++m)
        for (int n = 0; n <= m; ++n)
          vs[pos++] = L(m,n).vi_;
      for (int m = 0, mpos=0; m < K; ++m, mpos += m) {
        // FIXME: replace with dot_self
        LLt(m,m) = var(new dot_self_vari(vs + mpos, m + 1));
        for (int n = 0, npos = 0; n < m; ++n, npos += n)
          LLt(m,n) = LLt(n,m) = var(new dot_product_vv_vari(vs + mpos, vs + npos, n + 1));
      }
      return LLt;
    }

    /**
     * Returns the result of post-multiplying a matrix by its
     * own transpose.
     * @param M Matrix to multiply.
     * @return M times its transpose.
     */
    inline matrix_v
    tcrossprod(const matrix_v& M) {
      if(M.rows() == 0)
        return matrix_v(0,0);
      if(M.rows() == 1) {
        return M * M.transpose(); // FIXME: replace with dot_self
      }
      matrix_v result(M.rows(),M.rows());
      return result.setZero().selfadjointView<Eigen::Upper>().rankUpdate(M);
    }

    /**
     * Returns the result of pre-multiplying a matrix by its
     * own transpose.
     * @param M Matrix to multiply.
     * @return Transpose of M times M
     */
    inline matrix_v
    crossprod(const matrix_v& M) {
      return tcrossprod(M.transpose());
    }

    /**
     * Return the specified row minus 1 of the specified matrix.  That is,
     * indexing is from 1, not 0, so this function returns the same
     * value as the Eigen matrix call <code>m.row(i - 1)</code>.
     * @param[in] m Matrix.
     * @param[in] i Row index (plus 1).
     * @return Specified row of the matrix; between 1 and the number
     * of rows of <code>m</code> inclusive.
     * @throws std::domain_error If the index is 0 or
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
     * @throws std::domain_error if the index is 0 or greater than
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
     * @param v Vector to transform
     * @return Unit simplex result of the softmax transform of the vector.
     */
    vector_v softmax(const vector_v& v);

    template<int R1,int C1,int R2,int C2>
    inline Eigen::Matrix<var,R1,C2> mdivide_left_tri_low(const Eigen::Matrix<var,R1,C1> &A,
                                                         const Eigen::Matrix<var,R2,C2> &b) {
      stan::math::validate_square(A,"mdivide_left_tri_low");
      stan::math::validate_multiplicable(A,b,"mdivide_left_tri_low");
      return A.template triangularView<Eigen::Lower>().solve(b);
    }
    // FIXME: fold next two into above by templating out scalar types
    // & emplying to_var to both variables
    template<int R1,int C1,int R2,int C2>
    inline Eigen::Matrix<var,R1,C2> mdivide_left_tri_low(const Eigen::Matrix<var,R1,C1> &A,
                                                         const Eigen::Matrix<double,R2,C2> &b) {
      stan::math::validate_square(A,"mdivide_left_tri_low");
      stan::math::validate_multiplicable(A,b,"mdivide_left_tri_low");
      return A.template triangularView<Eigen::Lower>().solve(to_var(b));
    }
    template<int R1,int C1,int R2,int C2>
    inline Eigen::Matrix<var,R1,C2> mdivide_left_tri_low(const Eigen::Matrix<double,R1,C1> &A,
                                                         const Eigen::Matrix<var,R2,C2> &b) {
      stan::math::validate_square(A,"mdivide_left_tri_low");
      stan::math::validate_multiplicable(A,b,"mdivide_left_tri_low");
      return to_var(A).template triangularView<Eigen::Lower>().solve(b);
    }

    inline Eigen::Matrix<var,Eigen::Dynamic,Eigen::Dynamic>
    mdivide_left_tri_low(const Eigen::Matrix<var,Eigen::Dynamic,Eigen::Dynamic> &A) {
      stan::math::validate_square(A,"mdivide_left_tri_low");
      int n = A.rows();
      Eigen::Matrix<var,Eigen::Dynamic,Eigen::Dynamic> b;
      b.setIdentity(n,n);
      A.triangularView<Eigen::Lower>().solveInPlace(b);
      return(b);
    }

    /**
     * Returns the solution of the system Ax=b when A is triangular.
     * @param[in] A Triangular matrix.  Upper or lower is defined by TriView being
     * either Eigen::Upper or Eigen::Lower.
     * @param[in] b Right hand side matrix or vector.
     * @return x = A^-1 b, solution of the linear system.
     * @throws std::domain_error if A is not square or the rows of b don't
     * match the size of A.
     */
    template<int TriView,int R1,int C1,int R2,int C2>
    inline Eigen::Matrix<var,R1,C2> mdivide_left_tri(const Eigen::Matrix<var,R1,C1> &A,
                                                     const Eigen::Matrix<var,R2,C2> &b) {
      stan::math::validate_square(A,"mdivide_left_tri");
      stan::math::validate_multiplicable(A,b,"mdivide_left_tri");
      return A.template triangularView<TriView>().solve(b);
    }
    /**
     * Returns the solution of the system Ax=b when A is triangular.
     * @param[in] A Triangular matrix.  Upper or lower is defined by TriView being
     * either Eigen::Upper or Eigen::Lower.
     * @param[in] b Right hand side matrix or vector.
     * @return x = A^-1 b, solution of the linear system.
     * @throws std::domain_error if A is not square or the rows of b don't
     * match the size of A.
     */
    template<int TriView,int R1,int C1,int R2,int C2>
    inline Eigen::Matrix<var,R1,C2> mdivide_left_tri(const Eigen::Matrix<var,R1,C1> &A,
                                                     const Eigen::Matrix<double,R2,C2> &b) {
      stan::math::validate_square(A,"mdivide_left_tri");
      stan::math::validate_multiplicable(A,b,"mdivide_left_tri");
      return A.template triangularView<TriView>().solve(to_var(b));
    }
    /**
     * Returns the solution of the system Ax=b when A is triangular.
     * @param[in] A Triangular matrix.  Upper or lower is defined by TriView being
     * either Eigen::Upper or Eigen::Lower.
     * @param[in] b Right hand side matrix or vector.
     * @return x = A^-1 b, solution of the linear system.
     * @throws std::domain_error if A is not square or the rows of b don't
     * match the size of A.
     */
    template<int TriView,int R1,int C1,int R2,int C2>
    inline Eigen::Matrix<var,R1,C2> mdivide_left_tri(const Eigen::Matrix<double,R1,C1> &A,
                                                     const Eigen::Matrix<var,R2,C2> &b) {
      stan::math::validate_square(A,"mdivide_left_tri");
      stan::math::validate_multiplicable(A,b,"mdivide_left_tri");
      return to_var(A).template triangularView<TriView>().solve(b);
    }

    /**
     * Returns the solution of the system Ax=b when A is triangular and b = I.
     * @param[in] A Triangular matrix.  Upper or lower is defined by TriView being
     * either Eigen::Upper or Eigen::Lower.
     * @return x = A^-1 .
     * @throws std::domain_error if A is not square
     */
    template<int TriView>
    inline Eigen::Matrix<var,Eigen::Dynamic,Eigen::Dynamic>
    mdivide_left_tri(const Eigen::Matrix<var,Eigen::Dynamic,Eigen::Dynamic> &A) {
      stan::math::validate_square(A,"mdivide_left_tri_low");
      int n = A.rows();
      Eigen::Matrix<var,Eigen::Dynamic,Eigen::Dynamic> b;
      b.setIdentity(n,n);
      A.triangularView<TriView>().solveInPlace(b);
      return b;
    }

    /**
     * Returns the solution of the system Ax=b.
     * @param[in] A Matrix.
     * @param[in] b Right hand side matrix or vector.
     * @return x = A^-1 b, solution of the linear system.
     * @throws std::domain_error if A is not square or the rows of b don't
     * match the size of A.
     */
    template<int R1,int C1,int R2,int C2>
    inline Eigen::Matrix<var,R1,C2> mdivide_left(const Eigen::Matrix<var,R1,C1> &A,
                                                 const Eigen::Matrix<var,R2,C2> &b) {
      stan::math::validate_square(A,"mdivide_left");
      stan::math::validate_multiplicable(A,b,"mdivide_left");
      return A.lu().solve(b);
    }
    /**
     * Returns the solution of the system Ax=b.
     * @param[in] A Matrix.
     * @param[in] b Right hand side matrix or vector.
     * @return x = A^-1 b, solution of the linear system.
     * @throws std::domain_error if A is not square or the rows of b don't
     * match the size of A.
     */
    template<int R1, int C1, int R2, int C2>
    inline Eigen::Matrix<var,R1,C2> mdivide_left(const Eigen::Matrix<double,R1,C1> &A,
                                                 const Eigen::Matrix<var,R2,C2> &b) {
      stan::math::validate_square(A,"mdivide_left");
      stan::math::validate_multiplicable(A,b,"mdivide_left");
      // FIXME: it would be much faster to do LU, then convert to var
      return to_var(A).lu().solve(b);
    }
    /**
     * Returns the solution of the system Ax=b.
     * @param[in] A Matrix.
     * @param[in] b Right hand side matrix or vector.
     * @return x = A^-1 b, solution of the linear system.
     * @throws std::domain_error if A is not square or the rows of b don't
     * match the size of A.
     */
    template<int R1, int C1, int R2, int C2>
    inline Eigen::Matrix<var,R1,C2> mdivide_left(const Eigen::Matrix<var,R1,C1> &A,
                                                 const Eigen::Matrix<double,R2,C2> &b) {
      stan::math::validate_square(A,"mdivide_left");
      stan::math::validate_multiplicable(A,b,"mdivide_left");
      return A.lu().solve(to_var(b));
    }

    /**
     * Returns the solution x of the system xA = b.
     * @param[in] b Right hand side matrix or vector.
     * @param[in] A Matrix.
     * @return x = b A^-1, solution of the linear system.
     * @throws std::domain_error if A is not square or the cols of b don't
     * match the size of A.
     */
    template<int R1,int C1,int R2,int C2>
    inline Eigen::Matrix<var,R1,C2> mdivide_right(const Eigen::Matrix<var,R1,C1> &b,
                                                  const Eigen::Matrix<var,R2,C2> &A) {
      stan::math::validate_square(A,"mdivide_right");
      stan::math::validate_multiplicable(b,A,"mdivide_right");
      return A.transpose().lu().solve(b.transpose()).transpose();
    }
    /**
     * Returns the solution x of the system xA = b.
     * @param[in] b Right hand side matrix or vector.
     * @param[in] A Matrix.
     * @return x = b A^-1, solution of the linear system.
     * @throws std::domain_error if A is not square or the cols of b don't
     * match the size of A.
     */
    template<int R1,int C1,int R2,int C2>
    inline Eigen::Matrix<var,R1,C2> mdivide_right(const Eigen::Matrix<double,R1,C1> &b,
                                                  const Eigen::Matrix<var,R2,C2> &A) {
      stan::math::validate_square(A,"mdivide_right");
      stan::math::validate_multiplicable(b,A,"mdivide_right");
      return A.transpose().lu().solve(to_var(b).transpose()).transpose();
    }
    /**
     * Returns the solution x of the system xA = b.
     * @param[in] b Right hand side matrix or vector.
     * @param[in] A Matrix.
     * @return x = b A^-1, solution of the linear system.
     * @throws std::domain_error if A is not square or the cols of b don't
     * match the size of A.
     */
    template<int R1,int C1,int R2,int C2>
    inline Eigen::Matrix<var,R1,C2> mdivide_right(const Eigen::Matrix<var,R1,C1> &b,
                                                  const Eigen::Matrix<double,R2,C2> &A) {
      stan::math::validate_square(A,"mdivide_right");
      stan::math::validate_multiplicable(b,A,"mdivide_right");
      return to_var(A).transpose().lu().solve(b.transpose()).transpose();
    }

   /**
     * Returns the solution of the system Ax=b when A is triangular
     * @param A matrix.  
     * @param b Right hand side matrix or vector.
     * @return x = A^-1 b, solution of the linear system.
     * @tparam TriView triangular view of data, Eigen::Upper or Eigen::Lower
     * @throws std::domain_error if A is not square or the rows of b don't
     * match the size of A.
     */
    template<int TriView,typename T1,typename T2,int R1,int C1,int R2,int C2>
    inline Eigen::Matrix<typename boost::math::tools::promote_args<T1,T2>::type,R1,C2>
    mdivide_right_tri(const Eigen::Matrix<T1,R1,C1> &b,
                      const Eigen::Matrix<T2,R2,C2> &A) {
      stan::math::validate_square(A,"mdivide_right_tri_low");
      stan::math::validate_multiplicable(b,A,"mdivide_right_tri_low");
      return to_var(A)
        .template triangularView<TriView>()
        .transpose()
        .solve(to_var(b).transpose())
        .transpose();
    }
   /**
     * Returns the solution of the system tri(A)x=b when tri(A) is a
     * triangular view of A.
     * @param A Matrix.  Specify upper or lower with TriView
     * being Eigen::Upper or Eigen::Lower.
     * @param b Right hand side matrix or vector.
     * @return x = A^-1 b, solution of the linear system.
     * @throws std::domain_error if A is not square or the rows of b don't
     * match the size of A.
     */
    template<typename T1,typename T2,int R1,int C1,int R2,int C2>
    inline Eigen::Matrix<typename boost::math::tools::promote_args<T1,T2>::type,R1,C2>
    mdivide_right_tri_low(const Eigen::Matrix<T1,R1,C1> &b,
                          const Eigen::Matrix<T2,R2,C2> &A) {
      return mdivide_right_tri<Eigen::Lower,T1,T2,R1,C1,R2,C2>(b,A);
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


    // FIXME:  double val?
    inline void assign_to_var(stan::agrad::var& var, const double& val) {
      var = val;
    }
    inline void assign_to_var(stan::agrad::var& var, const stan::agrad::var& val) {
      var = val;
    }
    // FIXME:  int val?
    inline void assign_to_var(int& n_lhs, const int& n_rhs) {
      n_lhs = n_rhs;  // FIXME: no call -- just filler to instantiate
    }
    // FIXME:  double val?
    inline void assign_to_var(double& n_lhs, const double& n_rhs) {
      n_lhs = n_rhs;  // FIXME: no call -- just filler to instantiate
    }

    template <typename LHS, typename RHS>
    inline void assign_to_var(std::vector<LHS>& x, const std::vector<RHS>& y) {
      for (size_t i = 0; i < x.size(); ++i)
        assign_to_var(x[i],y[i]);
    }
    template <typename LHS, typename RHS>
    inline void assign_to_var(Eigen::Matrix<LHS,Eigen::Dynamic,1>& x, 
                              const Eigen::Matrix<RHS,Eigen::Dynamic,1>& y) {
      for (size_t i = 0; i < x.size(); ++i)
        assign_to_var(x(i),y(i));
    }
    template <typename LHS, typename RHS>
    inline void assign_to_var(Eigen::Matrix<LHS,1,Eigen::Dynamic>& x, 
                              const Eigen::Matrix<RHS,1,Eigen::Dynamic>& y) {
      for (size_t i = 0; i < x.size(); ++i)
        assign_to_var(x(i),y(i));
    }
    template <typename LHS, typename RHS>
    inline void assign_to_var(Eigen::Matrix<LHS,Eigen::Dynamic,Eigen::Dynamic>& x, 
                      const Eigen::Matrix<RHS,Eigen::Dynamic,Eigen::Dynamic>& y) {
      for (size_t n = 0; n < x.cols(); ++n)
        for (size_t m = 0; m < x.rows(); ++m)
          assign_to_var(x(m,n),y(m,n));
    }

    
    template <typename LHS, typename RHS>
    struct needs_promotion {
      enum { value = ( is_constant_struct<RHS>::value 
                       && !is_constant_struct<LHS>::value) };
    };

    template <bool PromoteRHS, typename LHS, typename RHS>
    struct assigner {
      static inline void assign(LHS& var, const RHS& val) {
        throw std::domain_error("should not call base class of assigner");
      }
    };

    template <typename LHS, typename RHS>
    struct assigner<false,LHS,RHS> {
      static inline void assign(LHS& var, const RHS& val) {
        var = val; // no promotion of RHS
      }
    };

    template <typename LHS, typename RHS>
    struct assigner<true,LHS,RHS> {
      static inline void assign(LHS& var, const RHS& val) {
        assign_to_var(var,val); // promote RHS
      }
    };

    
    template <typename LHS, typename RHS>
    inline void assign(LHS& var, const RHS& val) {
      assigner<needs_promotion<LHS,RHS>::value, LHS, RHS>::assign(var,val);
    }

    void stan_print(std::ostream* o, const var& x);

  }
}


#endif

