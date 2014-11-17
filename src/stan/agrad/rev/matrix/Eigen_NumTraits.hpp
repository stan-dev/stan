#ifndef STAN__AGRAD__REV__MATRIX__EIGEN_NUMTRAITS_HPP
#define STAN__AGRAD__REV__MATRIX__EIGEN_NUMTRAITS_HPP

#include <limits>
#include <stan/agrad/rev/var.hpp>
#include <stan/math/matrix/Eigen.hpp>

namespace stan {
  namespace agrad {
    
    namespace {
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
        virtual ~gevv_vvv_vari() {}
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
}


namespace Eigen {

  /**
   * Numerical traits template override for Eigen for automatic
   * gradient variables.
   */
  template <>
  struct NumTraits<stan::agrad::var> {
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
     * point, <code>std::numeric_limits<double>::epsilon()</code>.
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
        using std::log;
        return cast<double,int>(ceil(-log(std::numeric_limits<double>::epsilon())
                                     / log(10.0)));
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

#endif
