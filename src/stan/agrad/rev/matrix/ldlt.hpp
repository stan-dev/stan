#ifndef __STAN__AGRAD__REV__MATRIX__LDLT_HPP__
#define __STAN__AGRAD__REV__MATRIX__LDLT_HPP__

#include <vector>
#include <boost/type_traits.hpp>
#include <boost/utility/enable_if.hpp>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/math/matrix/validate_multiplicable.hpp>
#include <stan/math/matrix/validate_square.hpp>
#include <stan/math/matrix/ldlt.hpp>
#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/matrix/typedefs.hpp>

namespace stan {
  namespace agrad {
    namespace {
      template<int R, int C>
      class LDLT_alloc : public chainable_alloc {
      public:
        LDLT_alloc() : _N(0) {}
        LDLT_alloc(const Eigen::Matrix<var,R,C> &A) : _N(0) {
          compute(A);
        }
        
        inline void compute(const Eigen::Matrix<var,R,C> &A) {
          Eigen::Matrix<double,R,C> Ad(A.rows(),A.cols());

          _N = A.rows();
          _variA.resize(A.rows(),A.cols());

          for (size_t j = 0; j < _N; j++) {
            for (size_t i = 0; i < _N; i++) {
              Ad(i,j) = A(i,j).val();
              _variA(i,j) = A(i,j).vi_;
            }
          }
          
          _ldlt.compute(Ad);
        }
        inline double log_abs_det() const {
          return _ldlt.vectorD().array().log().sum();
        }
        
        size_t _N;
        Eigen::LDLT< Eigen::Matrix<double,R,C> > _ldlt;
        Eigen::Matrix<vari*,R,C> _variA;
      };
    }
  }
}

namespace stan {
  namespace math {
    template<int R, int C>
    class LDLT_factor<stan::agrad::var,R,C> {
    public:
      LDLT_factor() : _alloc(new stan::agrad::LDLT_alloc<R,C>()) {}
      LDLT_factor(const Eigen::Matrix<stan::agrad::var,R,C> &A) : _alloc(new stan::agrad::LDLT_alloc<R,C>(A)) { }
      
      inline void compute(const Eigen::Matrix<stan::agrad::var,R,C> &A) {
        stan::math::validate_square(A,"LDLT_factor<var>::compute");
        _alloc->compute(A);
      }
      

      template<typename Rhs>
      inline const Eigen::internal::solve_retval<Eigen::LDLT< Eigen::Matrix<double,R,C> >, Rhs>
      solve(const Eigen::MatrixBase<Rhs>& b) const {
        return _alloc->_ldlt.solve(b);
      }
      
      inline bool success() const {
        bool ret;
        ret = _alloc->_ldlt.info() == Eigen::Success;
        ret = ret && _alloc->_ldlt.isPositive();
        ret = ret && (_alloc->_ldlt.vectorD().array() > 0).all();
        return ret;
      }
      
      inline size_t rows() const { return _alloc->_N; }
      inline size_t cols() const { return _alloc->_N; }
      
      typedef size_t size_type;

      stan::agrad::LDLT_alloc<R,C> *_alloc;
    };
  }
}

namespace stan {
  namespace agrad {
    namespace {
      template<int R,int C>
      class log_det_ldlt_vari : public vari {
      public:
        log_det_ldlt_vari(const stan::math::LDLT_factor<var,R,C> &A)
        : vari(A._alloc->log_abs_det()), _alloc_ldlt(A._alloc)
        { }

        virtual void chain() {
          Eigen::Matrix<double,R,C> invA;
          
          // If we start computing Jacobians, this may be a bit inefficient
          invA.setIdentity(_alloc_ldlt->_N, _alloc_ldlt->_N);
          _alloc_ldlt->_ldlt.solveInPlace(invA);

          for (size_t j = 0; j < _alloc_ldlt->_N; j++) {
            for (size_t i = 0; i < _alloc_ldlt->_N; i++) {
              _alloc_ldlt->_variA(i,j)->adj_ += adj_ * invA(i,j);
            }
          }
        }
        
        const LDLT_alloc<R,C> *_alloc_ldlt;
      };

      template <int R1,int C1,int R2,int C2>
      class mdivide_left_ldlt_alloc : public chainable_alloc {
      public:
        virtual ~mdivide_left_ldlt_alloc() {}
        
        boost::shared_ptr< Eigen::LDLT< Eigen::Matrix<double,R1,C1> > > _ldltP;
        Eigen::Matrix<double,R2,C2> _C;
      };
      
      template <int R1,int C1,int R2,int C2>
      class mdivide_left_ldlt_vv_vari : public vari {
      public:
        int _M; // A.rows() = A.cols() = B.rows()
        int _N; // B.cols()
        vari** _variRefB;
        vari** _variRefC;
        mdivide_left_ldlt_alloc<R1,C1,R2,C2> *_alloc;
        const LDLT_alloc<R1,C1> *_alloc_ldlt;
        
        mdivide_left_ldlt_vv_vari(const stan::math::LDLT_factor<var,R1,C1> &A,
                                  const Eigen::Matrix<var,R2,C2> &B)
        : vari(0.0),
        _M(A.rows()),
        _N(B.cols()),
        _variRefB((vari**)stan::agrad::memalloc_.alloc(sizeof(vari*) 
                                                       * B.rows() * B.cols())),
        _variRefC((vari**)stan::agrad::memalloc_.alloc(sizeof(vari*) 
                                                       * B.rows() * B.cols())),
        _alloc(new mdivide_left_ldlt_alloc<R1,C1,R2,C2>()),
        _alloc_ldlt(A._alloc)
        {
          size_t pos = 0;
          _alloc->_C.resize(_M,_N);
          for (size_type j = 0; j < _N; j++) {
            for (size_type i = 0; i < _M; i++) {
              _variRefB[pos] = B(i,j).vi_;
              _alloc->_C(i,j) = B(i,j).val();
              pos++;
            }
          }
          
          _alloc_ldlt->_ldlt.solveInPlace(_alloc->_C);
          
          pos = 0;
          for (size_type j = 0; j < _N; j++) {
            for (size_type i = 0; i < _M; i++) {
              _variRefC[pos] = new vari(_alloc->_C(i,j),false);
              pos++;
            }
          }
        }
        
        virtual void chain() {
          Eigen::Matrix<double,R1,C1> adjA(_M,_M);
          Eigen::Matrix<double,R2,C2> adjB(_M,_N);
          
          size_t pos = 0;
          for (size_type j = 0; j < _N; j++)
            for (size_type i = 0; i < _M; i++)
              adjB(i,j) = _variRefC[pos++]->adj_;
          
          _alloc_ldlt->_ldlt.solveInPlace(adjB);
          adjA.noalias() = -adjB * _alloc->_C.transpose();

          for (size_type j = 0; j < _M; j++)
            for (size_type i = 0; i < _M; i++)
              _alloc_ldlt->_variA(i,j)->adj_ += adjA(i,j);
          
          pos = 0;
          for (size_type j = 0; j < _N; j++)
            for (size_type i = 0; i < _M; i++)
              _variRefB[pos++]->adj_ += adjB(i,j);
        }
      };

      template <int R1,int C1,int R2,int C2>
      class mdivide_left_ldlt_dv_vari : public vari {
      public:
        int _M; // A.rows() = A.cols() = B.rows()
        int _N; // B.cols()
        vari** _variRefB;
        vari** _variRefC;
        mdivide_left_ldlt_alloc<R1,C1,R2,C2> *_alloc;
        
        mdivide_left_ldlt_dv_vari(const stan::math::LDLT_factor<double,R1,C1> &A,
                                  const Eigen::Matrix<var,R2,C2> &B)
        : vari(0.0),
        _M(A.rows()),
        _N(B.cols()),
        _variRefB((vari**)stan::agrad::memalloc_.alloc(sizeof(vari*) 
                                                       * B.rows() * B.cols())),
        _variRefC((vari**)stan::agrad::memalloc_.alloc(sizeof(vari*) 
                                                       * B.rows() * B.cols())),
        _alloc(new mdivide_left_ldlt_alloc<R1,C1,R2,C2>())
        {
          using Eigen::Matrix;
          using Eigen::Map;
          
          size_t pos = 0;
          _alloc->_C.resize(_M,_N);
          for (size_type j = 0; j < _N; j++) {
            for (size_type i = 0; i < _M; i++) {
              _variRefB[pos] = B(i,j).vi_;
              _alloc->_C(i,j) = B(i,j).val();
              pos++;
            }
          }
          
          _alloc->_ldltP = A._ldltP;
          _alloc->_ldltP->solveInPlace(_alloc->_C);
          
          pos = 0;
          for (size_type j = 0; j < _N; j++) {
            for (size_type i = 0; i < _M; i++) {
              _variRefC[pos] = new vari(_alloc->_C(i,j),false);
              pos++;
            }
          }
        }
        
        virtual void chain() {
          Eigen::Matrix<double,R2,C2> adjB(_M,_N);
          
          size_t pos = 0;
          for (size_type j = 0; j < adjB.cols(); j++)
            for (size_type i = 0; i < adjB.rows(); i++)
              adjB(i,j) = _variRefC[pos++]->adj_;
          
          _alloc->_ldltP->solveInPlace(adjB);
          
          pos = 0;
          for (size_type j = 0; j < adjB.cols(); j++)
            for (size_type i = 0; i < adjB.rows(); i++)
              _variRefB[pos++]->adj_ += adjB(i,j);
        }
      };

      template <int R1,int C1,int R2,int C2>
      class mdivide_left_ldlt_vd_vari : public vari {
      public:
        int _M; // A.rows() = A.cols() = B.rows()
        int _N; // B.cols()
        vari** _variRefC;
        mdivide_left_ldlt_alloc<R1,C1,R2,C2> *_alloc;
        const LDLT_alloc<R1,C1> *_alloc_ldlt;
      
        mdivide_left_ldlt_vd_vari(const stan::math::LDLT_factor<var,R1,C1> &A,
                                  const Eigen::Matrix<double,R2,C2> &B)
          : vari(0.0),
            _M(A.rows()),
            _N(B.cols()),
            _variRefC((vari**)stan::agrad::memalloc_.alloc(sizeof(vari*) 
                                                           * B.rows() * B.cols())),
            _alloc(new mdivide_left_ldlt_alloc<R1,C1,R2,C2>()),
            _alloc_ldlt(A._alloc)
        {
          _alloc->_C = B;
          _alloc_ldlt->_ldlt.solveInPlace(_alloc->_C);
          
          size_t pos = 0;
          for (size_type j = 0; j < _N; j++) {
            for (size_type i = 0; i < _M; i++) {
              _variRefC[pos] = new vari(_alloc->_C(i,j),false);
              pos++;
            }
          }
        }
      
        virtual void chain() {
          Eigen::Matrix<double,R1,C1> adjA(_M,_M);
          Eigen::Matrix<double,R1,C2> adjC(_M,_N);

          size_t pos = 0;
          for (size_type j = 0; j < adjC.cols(); j++)
            for (size_type i = 0; i < adjC.rows(); i++)
              adjC(i,j) = _variRefC[pos++]->adj_;
        
          adjA = -_alloc_ldlt->_ldlt.solve(adjC*_alloc->_C.transpose());

          for (size_type j = 0; j < adjA.cols(); j++)
            for (size_type i = 0; i < adjA.rows(); i++)
              _alloc_ldlt->_variA(i,j)->adj_ += adjA(i,j);
        }
      };
    }

    template<int R, int C>
    var log_determinant_ldlt(stan::math::LDLT_factor<var,R,C> &A) {
      return var(new log_det_ldlt_vari<R,C>(A));
    }
    
    template <int R1,int C1,int R2,int C2>
    inline Eigen::Matrix<var,R1,C2>
    mdivide_left_ldlt(const stan::math::LDLT_factor<var,R1,C1> &A,
                      const Eigen::Matrix<var,R2,C2> &b) {
      Eigen::Matrix<var,R1,C2> res(b.rows(),b.cols());

      stan::math::validate_multiplicable(A,b,"mdivide_left_ldlt");
      
      mdivide_left_ldlt_vv_vari<R1,C1,R2,C2> *baseVari = new mdivide_left_ldlt_vv_vari<R1,C1,R2,C2>(A,b);
      
      size_t pos = 0;
      for (size_type j = 0; j < res.cols(); j++)
        for (size_type i = 0; i < res.rows(); i++)
          res(i,j).vi_ = baseVari->_variRefC[pos++];
      
      return res;
    }

    template <int R1,int C1,int R2,int C2>
    inline Eigen::Matrix<var,R1,C2>
    mdivide_left_ldlt(const stan::math::LDLT_factor<var,R1,C1> &A,
                      const Eigen::Matrix<double,R2,C2> &b) {
      Eigen::Matrix<var,R1,C2> res(b.rows(),b.cols());
      
      stan::math::validate_multiplicable(A,b,"mdivide_left_ldlt");
      
      mdivide_left_ldlt_vd_vari<R1,C1,R2,C2> *baseVari = new mdivide_left_ldlt_vd_vari<R1,C1,R2,C2>(A,b);
      
      size_t pos = 0;
      for (size_type j = 0; j < res.cols(); j++)
        for (size_type i = 0; i < res.rows(); i++)
          res(i,j).vi_ = baseVari->_variRefC[pos++];
      
      return res;
    }
    
    template <int R1,int C1,int R2,int C2>
    inline Eigen::Matrix<var,R1,C2>
    mdivide_left_ldlt(const stan::math::LDLT_factor<double,R1,C1> &A,
                      const Eigen::Matrix<var,R2,C2> &b) {
      Eigen::Matrix<var,R1,C2> res(b.rows(),b.cols());
      
      stan::math::validate_multiplicable(A,b,"mdivide_left_ldlt");
      
      mdivide_left_ldlt_dv_vari<R1,C1,R2,C2> *baseVari = new mdivide_left_ldlt_dv_vari<R1,C1,R2,C2>(A,b);
      
      size_t pos = 0;
      for (size_type j = 0; j < res.cols(); j++)
        for (size_type i = 0; i < res.rows(); i++)
          res(i,j).vi_ = baseVari->_variRefC[pos++];
      
      return res;
    }

    namespace {
      template <typename T2, int R2,int C2,typename T3,int R3,int C3>
      class trace_inv_quad_form_ldlt_impl : public chainable_alloc {
      protected:
        inline void initializeB(const Eigen::Matrix<var,R3,C3> &B,bool haveD) { 
          Eigen::Matrix<double,R3,C3> Bd(B.rows(),B.cols());
          _variB.resize(B.rows(),B.cols());
          for (size_t j = 0; j < B.cols(); j++) {
            for (size_t i = 0; i < B.rows(); i++) {
              _variB(i,j) = B(i,j).vi_;
              Bd(i,j) = B(i,j).val();
            }
          }
          _AinvB = _ldlt.solve(Bd);
          if (haveD) 
            _C.noalias() = Bd.transpose()*_AinvB;
          else
            _value = (Bd.transpose()*_AinvB).trace();
        }
        inline void initializeB(const Eigen::Matrix<double,R3,C3> &B,bool haveD) {
          _AinvB = _ldlt.solve(B);
          if (haveD) 
            _C.noalias() = B.transpose()*_AinvB;
          else
            _value = (B.transpose()*_AinvB).trace();
        }

        template<int R1,int C1>
        inline void initializeD(const Eigen::Matrix<var,R1,C1> &D) {
          _D.resize(D.rows(),D.cols());
          _variD.resize(D.rows(),D.cols());
          for (size_t j = 0; j < D.cols(); j++) {
            for (size_t i = 0; i < D.rows(); i++) {
              _variD(i,j) = D(i,j).vi_;
              _D(i,j) = D(i,j).val();
            }
          }
        }
        template<int R1,int C1>
        inline void initializeD(const Eigen::Matrix<double,R1,C1> &D) {
          _D = D;
        }

       public:
        template<typename T1, int R1,int C1>
        trace_inv_quad_form_ldlt_impl(const Eigen::Matrix<T1,R1,C1> &D,
                                      const stan::math::LDLT_factor<T2,R2,C2> &A,
                                      const Eigen::Matrix<T3,R3,C3> &B)
        : _Dtype(boost::is_same<T1,var>::value?1:0),
          _ldlt(A)
        {
          initializeB(B,true);
          initializeD(D);
          
          _value = (_D*_C).trace();
        }
        
        trace_inv_quad_form_ldlt_impl(const stan::math::LDLT_factor<T2,R2,C2> &A,
                                      const Eigen::Matrix<T3,R3,C3> &B)
        : _Dtype(2),
        _ldlt(A)
        {
          initializeB(B,false);
        }
        
        const int _Dtype; // 0 = double, 1 = var, 2 = missing
        stan::math::LDLT_factor<T2,R2,C2> _ldlt;
        Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> _D;
        Eigen::Matrix<vari*,Eigen::Dynamic,Eigen::Dynamic> _variD;
        Eigen::Matrix<vari*,R3,C3> _variB;
        Eigen::Matrix<double,R3,C3> _AinvB;
        Eigen::Matrix<double,C3,C3> _C;
        double _value;
      };

      template <typename T2,int R2,int C2,typename T3,int R3,int C3>
      class  trace_inv_quad_form_ldlt_vari : public vari {
      protected:
        static inline void chainA(const double &adj,
                                  trace_inv_quad_form_ldlt_impl<double,R2,C2,T3,R3,C3> *impl) {
        }
        static inline void chainB(const double &adj,
                                  trace_inv_quad_form_ldlt_impl<T2,R2,C2,double,R3,C3> *impl) {
        }
        
        static inline void chainA(const double &adj,
                                  trace_inv_quad_form_ldlt_impl<var,R2,C2,T3,R3,C3> *impl) {
          Eigen::Matrix<double,R2,C2> aA;
          if (impl->_Dtype != 2)
            aA.noalias() = -adj*(impl->_AinvB*impl->_D.transpose()*impl->_AinvB.transpose());
          else
            aA.noalias() = -adj*(impl->_AinvB*impl->_AinvB.transpose());
          for (size_type j = 0; j < aA.cols(); j++)
            for (size_type i = 0; i < aA.rows(); i++)
              impl->_ldlt._alloc->_variA(i,j)->adj_ += aA(i,j);
        }
        static inline void chainB(const double &adj,
                                  trace_inv_quad_form_ldlt_impl<T2,R2,C2,var,R3,C3> *impl) {
          Eigen::Matrix<double,R3,C3> aB;
          if (impl->_Dtype != 2)
            aB.noalias() = adj*impl->_AinvB*(impl->_D + impl->_D.transpose());
          else
            aB.noalias() = adj*impl->_AinvB;
          for (size_type j = 0; j < aB.cols(); j++)
            for (size_type i = 0; i < aB.rows(); i++)
              impl->_variB(i,j)->adj_ += aB(i,j);
        }
        
      public:
        trace_inv_quad_form_ldlt_vari(trace_inv_quad_form_ldlt_impl<T2,R2,C2,T3,R3,C3> *impl)
        : vari(impl->_value), _impl(impl)
        {}
        
        virtual void chain() {
          // F = trace(D * B' * inv(A) * B)
          // aA = -aF * inv(A') * B * D' * B' * inv(A')
          // aB = aF*(inv(A) * B * D + inv(A') * B * D')
          // aD = aF*(B' * inv(A) * B)
          chainA(adj_, _impl);
          
          chainB(adj_, _impl);
          
          if (_impl->_Dtype == 1) {
            for (size_type j = 0; j < _impl->_variD.cols(); j++)
              for (size_type i = 0; i < _impl->_variD.rows(); i++)
                _impl->_variD(i,j)->adj_ += adj_*_impl->_C(i,j);
          }
        }

        trace_inv_quad_form_ldlt_impl<T2,R2,C2,T3,R3,C3> *_impl;
      };
    }
    
    /**
     * Compute the trace of an inverse quadratic form.  I.E., this computes
     *       trace(D B^T A^-1 B)
     * where D is a square matrix and the LDLT_factor of A is provided.
     **/
    template <typename T1,int R1,int C1,typename T2,int R2,int C2,typename T3,int R3,int C3>
    inline typename
    boost::enable_if_c<boost::is_same<T1,var>::value || 
                       boost::is_same<T2,var>::value || 
                       boost::is_same<T3,var>::value, var>::type
    trace_inv_quad_form_ldlt(const Eigen::Matrix<T1,R1,C1> &D,
                             const stan::math::LDLT_factor<T2,R2,C2> &A,
                             const Eigen::Matrix<T3,R3,C3> &B)
    {
      stan::math::validate_square(D,"trace_inv_quad_form_ldlt");
      stan::math::validate_multiplicable(A,B,"trace_inv_quad_form_ldlt");
      stan::math::validate_multiplicable(B,D,"trace_inv_quad_form_ldlt");
      
      trace_inv_quad_form_ldlt_impl<T2,R2,C2,T3,R3,C3> *_impl = new trace_inv_quad_form_ldlt_impl<T2,R2,C2,T3,R3,C3>(D,A,B);
      
      return var(new trace_inv_quad_form_ldlt_vari<T2,R2,C2,T3,R3,C3>(_impl));
    }
    
    /**
     * Compute the trace of an inverse quadratic form.  I.E., this computes
     *       trace(B^T A^-1 B)
     * where the LDLT_factor of A is provided.
     **/
    template <typename T2,int R2,int C2,typename T3,int R3,int C3>
    inline typename
    boost::enable_if_c<boost::is_same<T2,var>::value ||
                       boost::is_same<T3,var>::value, var>::type
    trace_inv_quad_form_ldlt(const stan::math::LDLT_factor<T2,R2,C2> &A,
                             const Eigen::Matrix<T3,R3,C3> &B)
    {
      stan::math::validate_multiplicable(A,B,"trace_inv_quad_form_ldlt");
      
      trace_inv_quad_form_ldlt_impl<T2,R2,C2,T3,R3,C3> *_impl = new trace_inv_quad_form_ldlt_impl<T2,R2,C2,T3,R3,C3>(A,B);
      
      return var(new trace_inv_quad_form_ldlt_vari<T2,R2,C2,T3,R3,C3>(_impl));
    }
  }
}
#endif
