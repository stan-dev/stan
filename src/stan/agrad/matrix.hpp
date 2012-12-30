#ifndef __STAN__AGRAD__MATRIX_HPP__
#define __STAN__AGRAD__MATRIX_HPP__

#include <stan/math/matrix.hpp>
#include <stan/math/matrix_error_handling.hpp>

#include <stan/agrad/special_functions.hpp>

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
      HasFloatingPoint = 1
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
                  level3_blocking<LhsScalar,RhsScalar>& /* blocking */,
                  GemmParallelInfo<Index>* /* info = 0 */)
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
    typedef 
    Eigen::Matrix<var,Eigen::Dynamic,Eigen::Dynamic>
    matrix_v;

    /**
     * The type of a (column) vector holding <code>stan::agrad::var</code>
     * values.
     */
    typedef 
    Eigen::Matrix<var,Eigen::Dynamic,1>
    vector_v;

    /**
     * The type of a row vector holding <code>stan::agrad::var</code>
     * values.
     */
    typedef 
    Eigen::Matrix<var,1,Eigen::Dynamic>
    row_vector_v;

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
     * Returns a stan::agrad::var variable with the input value.     
     *
     * @param[in] rv A row vector with automatic differentiation variables
     * @return A row vector with automatic differentiation variables
     *    with values of rv.
     */
    inline row_vector_v to_var(const row_vector_v& rv) {
      return rv;
    }



    // scalar returns

    // /**
    //  * Determinant of the matrix.
    //  *
    //  * Returns the determinant of the specified
    //  * square matrix.
    //  *
    //  * @param m Specified matrix.
    //  * @return Determinant of the matrix.
    //  * @throw std::domain_error if m is not a square matrix
    //  */
    // var determinant(const Eigen::Matrix<var, Eigen::Dynamic, Eigen::Dynamic>& m);
    
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
//      stan::math::validate_square(L,"multiply_lower_tri_self_transpose");
      int K = L.rows();
      int J = L.cols();
      matrix_v LLt(K,K);
      if (K == 0) return LLt;
      // if (K == 1) {
      //   LLt(0,0) = L(0,0) * L(0,0);
      //   return LLt;
      // }
      int Knz;
      if (K >= J)
        Knz = (K-J)*J + (J * (J + 1)) / 2;
      else // if (K < J)
        Knz = (K * (K + 1)) / 2;
      vari** vs = (vari**)memalloc_.alloc( Knz * sizeof(vari*) );
      int pos = 0;
      for (int m = 0; m < K; ++m)
        for (int n = 0; n < ((J < (m+1))?J:(m+1)); ++n) {
          vs[pos++] = L(m,n).vi_;
        }
      for (int m = 0, mpos=0; m < K; ++m, mpos += (J < m)?J:m) {
        LLt(m,m) = var(new dot_self_vari(vs + mpos, (J < (m+1))?J:(m+1)));
        for (int n = 0, npos = 0; n < m; ++n, npos += (J < n)?J:n) {
          LLt(m,n) = LLt(n,m) = var(new dot_product_vv_vari(vs + mpos, vs + npos, (J < (n+1))?J:(n+1)));
        }
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
      if (M.rows() == 0)
        return matrix_v(0,0);
      if (M.rows() == 1)
        return M * M.transpose();

      // WAS JUST THIS
      // matrix_v result(M.rows(),M.rows());
      // return result.setZero().selfadjointView<Eigen::Upper>().rankUpdate(M);

      matrix_v MMt(M.rows(),M.rows());

      vari** vs 
        = (vari**)memalloc_.alloc((M.rows() * M.cols() ) * sizeof(vari*));
      int pos = 0;
      for (int m = 0; m < M.rows(); ++m)
        for (int n = 0; n < M.cols(); ++n)
          vs[pos++] = M(m,n).vi_;
      for (int m = 0; m < M.rows(); ++m)
        MMt(m,m) = var(new dot_self_vari(vs + m * M.cols(),M.cols()));
      for (int m = 0; m < M.rows(); ++m) {
        for (int n = 0; n < m; ++n) {
          MMt(m,n) = var(new dot_product_vv_vari(vs + m * M.cols(),
                                                 vs + n * M.cols(),
                                                 M.cols()));
          MMt(n,m) = MMt(m,n);
        }
      }
      return MMt;
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
      typedef 
        typename Eigen::Matrix<LHS,Eigen::Dynamic,1>::size_type 
        size_type;
      for (size_type i = 0; i < x.size(); ++i)
        assign_to_var(x(i),y(i));
    }
    template <typename LHS, typename RHS>
    inline void assign_to_var(Eigen::Matrix<LHS,1,Eigen::Dynamic>& x, 
                              const Eigen::Matrix<RHS,1,Eigen::Dynamic>& y) {
      typedef 
        typename Eigen::Matrix<LHS,1,Eigen::Dynamic>::size_type 
        size_type;
      for (size_type i = 0; i < x.size(); ++i)
        assign_to_var(x(i),y(i));
    }
    template <typename LHS, typename RHS>
    inline void assign_to_var(Eigen::Matrix<LHS,Eigen::Dynamic,Eigen::Dynamic>& x, 
                      const Eigen::Matrix<RHS,Eigen::Dynamic,Eigen::Dynamic>& y) {
      typedef 
        typename Eigen::Matrix<LHS,Eigen::Dynamic,Eigen::Dynamic>::size_type 
        size_type1;
      typedef 
        typename Eigen::Matrix<RHS,Eigen::Dynamic,Eigen::Dynamic>::size_type 
        size_type2;
      for (size_type1 n = 0; n < x.cols(); ++n)
        for (size_type2 m = 0; m < x.rows(); ++m)
          assign_to_var(x(m,n),y(m,n));
    }

    
    template <typename LHS, typename RHS>
    struct needs_promotion {
      enum { value = ( is_constant_struct<RHS>::value 
                       && !is_constant_struct<LHS>::value) };
    };
    
    template <bool PromoteRHS, typename LHS, typename RHS>
    struct assigner {
      static inline void assign(LHS& /*var*/, const RHS& /*val*/) {
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

    void stan_print(std::ostream* o, const var& x) {
      *o << x.val();
    }

  }
}


#endif

