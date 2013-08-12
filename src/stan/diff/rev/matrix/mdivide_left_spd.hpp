#ifndef __STAN__AGRAD__REV__MATRIX__MDIVIDE_LEFT_SPD_HPP__
#define __STAN__AGRAD__REV__MATRIX__MDIVIDE_LEFT_SPD_HPP__

#include <vector>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/math/matrix/validate_multiplicable.hpp>
#include <stan/math/matrix/validate_square.hpp>
#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/matrix/typedefs.hpp>

namespace stan {
  namespace agrad {

    namespace {
      template <int R1,int C1,int R2,int C2>
      class mdivide_left_spd_alloc : public chainable_alloc {
      public:
        virtual ~mdivide_left_spd_alloc() {}

        Eigen::LLT< Eigen::Matrix<double,R1,C1> > _llt;
        Eigen::Matrix<double,R2,C2> _C;
      };
      
      template <int R1,int C1,int R2,int C2>
      class mdivide_left_spd_vv_vari : public vari {
      public:
        int _M; // A.rows() = A.cols() = B.rows()
        int _N; // B.cols()
        vari** _variRefA;
        vari** _variRefB;
        vari** _variRefC;
        mdivide_left_spd_alloc<R1,C1,R2,C2> *_alloc;

        mdivide_left_spd_vv_vari(const Eigen::Matrix<var,R1,C1> &A,
                                 const Eigen::Matrix<var,R2,C2> &B)
          : vari(0.0),
            _M(A.rows()),
            _N(B.cols()),
            _variRefA((vari**)stan::agrad::memalloc_.alloc(sizeof(vari*) 
                                                           * A.rows() * A.cols())),
            _variRefB((vari**)stan::agrad::memalloc_.alloc(sizeof(vari*) 
                                                           * B.rows() * B.cols())),
            _variRefC((vari**)stan::agrad::memalloc_.alloc(sizeof(vari*) 
                                                           * B.rows() * B.cols())),
            _alloc(new mdivide_left_spd_alloc<R1,C1,R2,C2>())
        {
          using Eigen::Matrix;
          using Eigen::Map;

          Matrix<double,R1,C1> Ad(A.rows(),A.cols());
          
          size_t pos = 0;
          for (size_type j = 0; j < _M; j++) {
            for (size_type i = 0; i < _M; i++) {
              _variRefA[pos] = A(i,j).vi_;
              Ad(i,j) = A(i,j).val();
              pos++;
            }
          }
  
          pos = 0;
          _alloc->_C.resize(_M,_N);
          for (size_type j = 0; j < _N; j++) {
            for (size_type i = 0; i < _M; i++) {
              _variRefB[pos] = B(i,j).vi_;
              _alloc->_C(i,j) = B(i,j).val();
              pos++;
            }
          }
        
          _alloc->_llt = Ad.llt();
          _alloc->_llt.solveInPlace(_alloc->_C);

          pos = 0;
          for (size_type j = 0; j < _N; j++) {
            for (size_type i = 0; i < _M; i++) {
              _variRefC[pos] = new vari(_alloc->_C(i,j),false);
              pos++;
            }
          }
        }
      
        virtual void chain() {
          using Eigen::Matrix;
          using Eigen::Map;
          Eigen::Matrix<double,R1,C1> adjA(_M,_M);
          Eigen::Matrix<double,R2,C2> adjB(_M,_N);

          size_t pos = 0;
          for (size_type j = 0; j < _N; j++)
            for (size_type i = 0; i < _M; i++)
              adjB(i,j) = _variRefC[pos++]->adj_;

          _alloc->_llt.solveInPlace(adjB);
          adjA.noalias() = -adjB * _alloc->_C.transpose();
        
          pos = 0;
          for (size_type j = 0; j < _M; j++)
            for (size_type i = 0; i < _M; i++)
              _variRefA[pos++]->adj_ += adjA(i,j);
        
          pos = 0;
          for (size_type j = 0; j < _N; j++)
            for (size_type i = 0; i < _M; i++)
              _variRefB[pos++]->adj_ += adjB(i,j);
        }
      };
    
      template <int R1,int C1,int R2,int C2>
      class mdivide_left_spd_dv_vari : public vari {
      public:
        int _M; // A.rows() = A.cols() = B.rows()
        int _N; // B.cols()
        vari** _variRefB;
        vari** _variRefC;
        mdivide_left_spd_alloc<R1,C1,R2,C2> *_alloc;
      
        mdivide_left_spd_dv_vari(const Eigen::Matrix<double,R1,C1> &A,
                                 const Eigen::Matrix<var,R2,C2> &B)
          : vari(0.0),
            _M(A.rows()),
            _N(B.cols()),
            _variRefB((vari**)stan::agrad::memalloc_.alloc(sizeof(vari*) 
                                                           * B.rows() * B.cols())),
            _variRefC((vari**)stan::agrad::memalloc_.alloc(sizeof(vari*) 
                                                           * B.rows() * B.cols())),
            _alloc(new mdivide_left_spd_alloc<R1,C1,R2,C2>())
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

          _alloc->_llt = A.llt();
          _alloc->_llt.solveInPlace(_alloc->_C);
          
          pos = 0;
          for (size_type j = 0; j < _N; j++) {
            for (size_type i = 0; i < _M; i++) {
              _variRefC[pos] = new vari(_alloc->_C(i,j),false);
              pos++;
            }
          }
        }
      
        virtual void chain() {
          using Eigen::Matrix;
          using Eigen::Map;
          Eigen::Matrix<double,R2,C2> adjB(_M,_N);

          size_t pos = 0;
          for (size_type j = 0; j < adjB.cols(); j++)
            for (size_type i = 0; i < adjB.rows(); i++)
              adjB(i,j) = _variRefC[pos++]->adj_;

          _alloc->_llt.solveInPlace(adjB);

          pos = 0;
          for (size_type j = 0; j < adjB.cols(); j++)
            for (size_type i = 0; i < adjB.rows(); i++)
              _variRefB[pos++]->adj_ += adjB(i,j);
        }
      };
    
      template <int R1,int C1,int R2,int C2>
      class mdivide_left_spd_vd_vari : public vari {
      public:
        int _M; // A.rows() = A.cols() = B.rows()
        int _N; // B.cols()
        vari** _variRefA;
        vari** _variRefC;
        mdivide_left_spd_alloc<R1,C1,R2,C2> *_alloc;
      
        mdivide_left_spd_vd_vari(const Eigen::Matrix<var,R1,C1> &A,
                                 const Eigen::Matrix<double,R2,C2> &B)
          : vari(0.0),
            _M(A.rows()),
            _N(B.cols()),
            _variRefA((vari**)stan::agrad::memalloc_.alloc(sizeof(vari*) 
                                                           * A.rows() * A.cols())),
            _variRefC((vari**)stan::agrad::memalloc_.alloc(sizeof(vari*) 
                                                           * B.rows() * B.cols())),
            _alloc(new mdivide_left_spd_alloc<R1,C1,R2,C2>())
        {
          using Eigen::Matrix;
          using Eigen::Map;

          Matrix<double,R1,C1> Ad(A.rows(),A.cols());
          
          size_t pos = 0;
          for (size_type j = 0; j < _M; j++) {
            for (size_type i = 0; i < _M; i++) {
              _variRefA[pos] = A(i,j).vi_;
              Ad(i,j) = A(i,j).val();
              pos++;
            }
          }
          
          _alloc->_llt = Ad.llt();
          _alloc->_C = _alloc->_llt.solve(B);
          
          pos = 0;
          for (size_type j = 0; j < _N; j++) {
            for (size_type i = 0; i < _M; i++) {
              _variRefC[pos] = new vari(_alloc->_C(i,j),false);
              pos++;
            }
          }
        }
      
        virtual void chain() {
          using Eigen::Matrix;
          using Eigen::Map;
          Eigen::Matrix<double,R1,C1> adjA(_M,_M);
          Eigen::Matrix<double,R1,C2> adjC(_M,_N);

          size_t pos = 0;
          for (size_type j = 0; j < adjC.cols(); j++)
            for (size_type i = 0; i < adjC.rows(); i++)
              adjC(i,j) = _variRefC[pos++]->adj_;
        
          adjA = -_alloc->_llt.solve(adjC*_alloc->_C.transpose());

          pos = 0;
          for (size_type j = 0; j < adjA.cols(); j++)
            for (size_type i = 0; i < adjA.rows(); i++)
              _variRefA[pos++]->adj_ += adjA(i,j);
        }
      };
    }

    template <int R1,int C1,int R2,int C2>
    inline 
    Eigen::Matrix<var,R1,C2>
    mdivide_left_spd(const Eigen::Matrix<var,R1,C1> &A,
                     const Eigen::Matrix<var,R2,C2> &b) {
      Eigen::Matrix<var,R1,C2> res(b.rows(),b.cols());
      
      stan::math::validate_square(A,"mdivide_left_spd");
      stan::math::validate_multiplicable(A,b,"mdivide_left_spd");
      
      // NOTE: this is not a memory leak, this vari is used in the 
      // expression graph to evaluate the adjoint, but is not needed
      // for the returned matrix.  Memory will be cleaned up with the arena allocator.
      mdivide_left_spd_vv_vari<R1,C1,R2,C2> *baseVari = new mdivide_left_spd_vv_vari<R1,C1,R2,C2>(A,b);
      
      size_t pos = 0;
      for (size_type j = 0; j < res.cols(); j++)
        for (size_type i = 0; i < res.rows(); i++)
          res(i,j).vi_ = baseVari->_variRefC[pos++];
      
      return res;
    }

    template <int R1,int C1,int R2,int C2>
    inline 
    Eigen::Matrix<var,R1,C2>
    mdivide_left_spd(const Eigen::Matrix<var,R1,C1> &A,
                     const Eigen::Matrix<double,R2,C2> &b) {
      Eigen::Matrix<var,R1,C2> res(b.rows(),b.cols());
      
      stan::math::validate_square(A,"mdivide_left_spd");
      stan::math::validate_multiplicable(A,b,"mdivide_left_spd");
      
      // NOTE: this is not a memory leak, this vari is used in the 
      // expression graph to evaluate the adjoint, but is not needed
      // for the returned matrix.  Memory will be cleaned up with the arena allocator.
      mdivide_left_spd_vd_vari<R1,C1,R2,C2> *baseVari = new mdivide_left_spd_vd_vari<R1,C1,R2,C2>(A,b);
      
      size_t pos = 0;
      for (size_type j = 0; j < res.cols(); j++)
        for (size_type i = 0; i < res.rows(); i++)
          res(i,j).vi_ = baseVari->_variRefC[pos++];
      
      return res;
    }
    
    template <int R1,int C1,int R2,int C2>
    inline 
    Eigen::Matrix<var,R1,C2>
    mdivide_left_spd(const Eigen::Matrix<double,R1,C1> &A,
                     const Eigen::Matrix<var,R2,C2> &b) {
      Eigen::Matrix<var,R1,C2> res(b.rows(),b.cols());
      
      stan::math::validate_square(A,"mdivide_left_spd");
      stan::math::validate_multiplicable(A,b,"mdivide_left_spd");
      
      // NOTE: this is not a memory leak, this vari is used in the 
      // expression graph to evaluate the adjoint, but is not needed
      // for the returned matrix.  Memory will be cleaned up with the arena allocator.
      mdivide_left_spd_dv_vari<R1,C1,R2,C2> *baseVari = new mdivide_left_spd_dv_vari<R1,C1,R2,C2>(A,b);
      
      size_t pos = 0;
      for (size_type j = 0; j < res.cols(); j++)
        for (size_type i = 0; i < res.rows(); i++)
          res(i,j).vi_ = baseVari->_variRefC[pos++];
      
      return res;
    }
    
  }
}
#endif
