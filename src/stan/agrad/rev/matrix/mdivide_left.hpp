#ifndef __STAN__AGRAD__REV__MATRIX__COLUMNS_MDIVIDE_LEFT_HPP__
#define __STAN__AGRAD__REV__MATRIX__COLUMNS_MDIVIDE_LEFT_HPP__

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
      class mdivide_left_vv_vari : public vari {
      public:
        int _M; // A.rows() = A.cols() = B.rows()
        int _N; // B.cols()
        double* _A;
        double* _C;
        vari** _variRefA;
        vari** _variRefB;
        vari** _variRefC;

        mdivide_left_vv_vari(const Eigen::Matrix<var,R1,C1> &A,
                             const Eigen::Matrix<var,R2,C2> &B)
          : vari(0.0),
            _M(A.rows()),
            _N(B.cols()),
            _A((double*)stan::agrad::memalloc_.alloc(sizeof(double) 
                                                     * A.rows() * A.cols())),
            _C((double*)stan::agrad::memalloc_.alloc(sizeof(double) 
                                                     * B.rows() * B.cols())),
            _variRefA((vari**)stan::agrad::memalloc_.alloc(sizeof(vari*) 
                                                           * A.rows() * A.cols())),
            _variRefB((vari**)stan::agrad::memalloc_.alloc(sizeof(vari*) 
                                                           * B.rows() * B.cols())),
            _variRefC((vari**)stan::agrad::memalloc_.alloc(sizeof(vari*) 
                                                           * B.rows() * B.cols()))
        {
          using Eigen::Matrix;
          using Eigen::Map;

          size_t pos = 0;
          for (size_type j = 0; j < _M; j++) {
            for (size_type i = 0; i < _M; i++) {
              _variRefA[pos] = A(i,j).vi_;
              _A[pos++] = A(i,j).val();
            }
          }
  
          pos = 0;
          for (size_type j = 0; j < _N; j++) {
            for (size_type i = 0; i < _M; i++) {
              _variRefB[pos] = B(i,j).vi_;
              _C[pos++] = B(i,j).val();
            }
          }
        
          Matrix<double,R1,C2> C(_M,_N);
          C = Map<Matrix<double,R1,C2> >(_C,_M,_N);

          C = Map<Matrix<double,R1,C1> >(_A,_M,_M)
            .colPivHouseholderQr().solve(C);

          pos = 0;
          for (size_type j = 0; j < _N; j++) {
            for (size_type i = 0; i < _M; i++) {
              _C[pos] = C(i,j);
              _variRefC[pos] = new vari(_C[pos],false);
              pos++;
            }
          }
        }
      
        virtual void chain() {
          using Eigen::Matrix;
          using Eigen::Map;
          Eigen::Matrix<double,R1,C1> adjA(_M,_M);
          Eigen::Matrix<double,R2,C2> adjB(_M,_N);
          Eigen::Matrix<double,R1,C2> adjC(_M,_N);

          size_t pos = 0;
          for (size_type j = 0; j < adjC.cols(); j++)
            for (size_type i = 0; i < adjC.rows(); i++)
              adjC(i,j) = _variRefC[pos++]->adj_;
        
        
          adjB = Map<Matrix<double,R1,C1> >(_A,_M,_M)
            .transpose().colPivHouseholderQr().solve(adjC);
          adjA.noalias() = -adjB
            * Map<Matrix<double,R1,C2> >(_C,_M,_N).transpose();
        
          pos = 0;
          for (size_type j = 0; j < adjA.cols(); j++)
            for (size_type i = 0; i < adjA.rows(); i++)
              _variRefA[pos++]->adj_ += adjA(i,j);
        
          pos = 0;
          for (size_type j = 0; j < adjB.cols(); j++)
            for (size_type i = 0; i < adjB.rows(); i++)
              _variRefB[pos++]->adj_ += adjB(i,j);
        }
      };
    
      template <int R1,int C1,int R2,int C2>
      class mdivide_left_dv_vari : public vari {
      public:
        int _M; // A.rows() = A.cols() = B.rows()
        int _N; // B.cols()
        double* _A;
        double* _C;
        vari** _variRefB;
        vari** _variRefC;
      
        mdivide_left_dv_vari(const Eigen::Matrix<double,R1,C1> &A,
                             const Eigen::Matrix<var,R2,C2> &B)
          : vari(0.0),
            _M(A.rows()),
            _N(B.cols()),
            _A((double*)stan::agrad::memalloc_.alloc(sizeof(double) 
                                                     * A.rows() * A.cols())),
            _C((double*)stan::agrad::memalloc_.alloc(sizeof(double) 
                                                     * B.rows() * B.cols())),
            _variRefB((vari**)stan::agrad::memalloc_.alloc(sizeof(vari*) 
                                                           * B.rows() * B.cols())),
            _variRefC((vari**)stan::agrad::memalloc_.alloc(sizeof(vari*) 
                                                           * B.rows() * B.cols()))
        {
          using Eigen::Matrix;
          using Eigen::Map;
  
          size_t pos = 0;
          for (size_type j = 0; j < _M; j++) {
            for (size_type i = 0; i < _M; i++) {
              _A[pos++] = A(i,j);
            }
          }
  
          pos = 0;
          for (size_type j = 0; j < _N; j++) {
            for (size_type i = 0; i < _M; i++) {
              _variRefB[pos] = B(i,j).vi_;
              _C[pos++] = B(i,j).val();
            }
          }
                
          Matrix<double,R1,C2> C(_M,_N);
          C = Map<Matrix<double,R1,C2> >(_C,_M,_N);

          C = Map<Matrix<double,R1,C1> >(_A,_M,_M)
            .colPivHouseholderQr().solve(C);
  
          pos = 0;
          for (size_type j = 0; j < _N; j++) {
            for (size_type i = 0; i < _M; i++) {
              _C[pos] = C(i,j);
              _variRefC[pos] = new vari(_C[pos],false);
              pos++;
            }
          }
        }
      
        virtual void chain() {
          using Eigen::Matrix;
          using Eigen::Map;
          Eigen::Matrix<double,R2,C2> adjB(_M,_N);
          Eigen::Matrix<double,R1,C2> adjC(_M,_N);

          size_t pos = 0;
          for (size_type j = 0; j < adjC.cols(); j++)
            for (size_type i = 0; i < adjC.rows(); i++)
              adjC(i,j) = _variRefC[pos++]->adj_;

          adjB = Map<Matrix<double,R1,C1> >(_A,_M,_M)
            .transpose().colPivHouseholderQr().solve(adjC);

          pos = 0;
          for (size_type j = 0; j < adjB.cols(); j++)
            for (size_type i = 0; i < adjB.rows(); i++)
              _variRefB[pos++]->adj_ += adjB(i,j);
        }
      };
    
      template <int R1,int C1,int R2,int C2>
      class mdivide_left_vd_vari : public vari {
      public:
        int _M; // A.rows() = A.cols() = B.rows()
        int _N; // B.cols()
        double* _A;
        double* _C;
        vari** _variRefA;
        vari** _variRefC;
      
        mdivide_left_vd_vari(const Eigen::Matrix<var,R1,C1> &A,
                             const Eigen::Matrix<double,R2,C2> &B)
          : vari(0.0),
            _M(A.rows()),
            _N(B.cols()),
            _A((double*)stan::agrad::memalloc_.alloc(sizeof(double) 
                                                     * A.rows() * A.cols())),
            _C((double*)stan::agrad::memalloc_.alloc(sizeof(double) 
                                                     * B.rows() * B.cols())),
            _variRefA((vari**)stan::agrad::memalloc_.alloc(sizeof(vari*) 
                                                           * A.rows() * A.cols())),
            _variRefC((vari**)stan::agrad::memalloc_.alloc(sizeof(vari*) 
                                                           * B.rows() * B.cols()))
        {
          using Eigen::Matrix;
          using Eigen::Map;

          size_t pos = 0;
          for (size_type j = 0; j < _M; j++) {
            for (size_type i = 0; i < _M; i++) {
              _variRefA[pos] = A(i,j).vi_;
              _A[pos++] = A(i,j).val();
            }
          }
  
          Matrix<double,R1,C2> C(_M,_N);
          C = Map<Matrix<double,R1,C1> >(_A,_M,_M)
            .colPivHouseholderQr().solve(B);

          pos = 0;
          for (size_type j = 0; j < _N; j++) {
            for (size_type i = 0; i < _M; i++) {
              _C[pos] = C(i,j);
              _variRefC[pos] = new vari(_C[pos],false);
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
        
          // FIXME: add .noalias() to LHS
          adjA = -Map<Matrix<double,R1,C1> >(_A,_M,_M)
            .transpose()
            .colPivHouseholderQr()
            .solve(adjC*Map<Matrix<double,R1,C2> >(_C,_M,_N).transpose());

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
    mdivide_left(const Eigen::Matrix<var,R1,C1> &A,
                 const Eigen::Matrix<var,R2,C2> &b) {
      Eigen::Matrix<var,R1,C2> res(b.rows(),b.cols());
      
      stan::math::validate_square(A,"mdivide_left");
      stan::math::validate_multiplicable(A,b,"mdivide_left");
      
      // NOTE: this is not a memory leak, this vari is used in the 
      // expression graph to evaluate the adjoint, but is not needed
      // for the returned matrix.  Memory will be cleaned up with the arena allocator.
      mdivide_left_vv_vari<R1,C1,R2,C2> *baseVari = new mdivide_left_vv_vari<R1,C1,R2,C2>(A,b);
      
      size_t pos = 0;
      for (size_type j = 0; j < res.cols(); j++)
        for (size_type i = 0; i < res.rows(); i++)
          res(i,j).vi_ = baseVari->_variRefC[pos++];
      
      return res;
    }

    template <int R1,int C1,int R2,int C2>
    inline 
    Eigen::Matrix<var,R1,C2>
    mdivide_left(const Eigen::Matrix<var,R1,C1> &A,
                 const Eigen::Matrix<double,R2,C2> &b) {
      Eigen::Matrix<var,R1,C2> res(b.rows(),b.cols());
      
      stan::math::validate_square(A,"mdivide_left");
      stan::math::validate_multiplicable(A,b,"mdivide_left");
      
      // NOTE: this is not a memory leak, this vari is used in the 
      // expression graph to evaluate the adjoint, but is not needed
      // for the returned matrix.  Memory will be cleaned up with the arena allocator.
      mdivide_left_vd_vari<R1,C1,R2,C2> *baseVari = new mdivide_left_vd_vari<R1,C1,R2,C2>(A,b);
      
      size_t pos = 0;
      for (size_type j = 0; j < res.cols(); j++)
        for (size_type i = 0; i < res.rows(); i++)
          res(i,j).vi_ = baseVari->_variRefC[pos++];
      
      return res;
    }
    
    template <int R1,int C1,int R2,int C2>
    inline 
    Eigen::Matrix<var,R1,C2>
    mdivide_left(const Eigen::Matrix<double,R1,C1> &A,
                 const Eigen::Matrix<var,R2,C2> &b) {
      Eigen::Matrix<var,R1,C2> res(b.rows(),b.cols());
      
      stan::math::validate_square(A,"mdivide_left");
      stan::math::validate_multiplicable(A,b,"mdivide_left");
      
      // NOTE: this is not a memory leak, this vari is used in the 
      // expression graph to evaluate the adjoint, but is not needed
      // for the returned matrix.  Memory will be cleaned up with the arena allocator.
      mdivide_left_dv_vari<R1,C1,R2,C2> *baseVari = new mdivide_left_dv_vari<R1,C1,R2,C2>(A,b);
      
      size_t pos = 0;
      for (size_type j = 0; j < res.cols(); j++)
        for (size_type i = 0; i < res.rows(); i++)
          res(i,j).vi_ = baseVari->_variRefC[pos++];
      
      return res;
    }
    
  }
}
#endif
