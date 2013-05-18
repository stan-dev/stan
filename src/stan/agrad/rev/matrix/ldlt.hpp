#ifndef __STAN__AGRAD__REV__MATRIX__LDLT_HPP__
#define __STAN__AGRAD__REV__MATRIX__LDLT_HPP__

#include <vector>
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
          
          _variARef = (vari**)stan::agrad::memalloc_.alloc(sizeof(vari*) 
                                                          * A.rows() * A.cols());
          _N = A.rows();
          
          size_t pos = 0;
          for (size_type j = 0; j < _N; j++) {
            for (size_type i = 0; i < _N; i++) {
              Ad(i,j) = A(i,j).val();
              _variARef[pos++] = A(i,j).vi_;
            }
          }
          
          _ldlt.compute(Ad);
        }
        inline double log_abs_det() const {
          return _ldlt.vectorD().array().log().sum();
        }
        
        size_t _N;
        Eigen::LDLT< Eigen::Matrix<double,R,C> > _ldlt;
        vari** _variARef;
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

          size_t pos = 0;
          for (size_type j = 0; j < _alloc_ldlt->_N; j++) {
            for (size_type i = 0; i < _alloc_ldlt->_N; i++) {
              _alloc_ldlt->_variARef[pos++]->adj_ += adj_ * invA(i,j);
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

          pos = 0;
          for (size_type j = 0; j < _M; j++)
            for (size_type i = 0; i < _M; i++)
              _alloc_ldlt->_variARef[pos++]->adj_ += adjA(i,j);
          
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

          pos = 0;
          for (size_type j = 0; j < adjA.cols(); j++)
            for (size_type i = 0; i < adjA.rows(); i++)
              _alloc_ldlt->_variARef[pos++]->adj_ += adjA(i,j);
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
  }
}
#endif
