#ifndef STAN__AGRAD__REV__MATRIX__MDIVIDE_LEFT_TRI_HPP
#define STAN__AGRAD__REV__MATRIX__MDIVIDE_LEFT_TRI_HPP

#include <vector>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/error_handling/matrix/check_multiplicable.hpp>
#include <stan/error_handling/matrix/check_square.hpp>
#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/matrix/typedefs.hpp>

namespace stan {
  namespace agrad {

    namespace {
      template <int TriView,int R1,int C1,int R2,int C2>
      class mdivide_left_tri_vv_vari : public vari {
      public:
        int M_; // A.rows() = A.cols() = B.rows()
        int N_; // B.cols()
        double* A_;
        double* C_;
        vari** _variRefA;
        vari** _variRefB;
        vari** _variRefC;
      
        mdivide_left_tri_vv_vari(const Eigen::Matrix<var,R1,C1> &A,
                                 const Eigen::Matrix<var,R2,C2> &B)
          : vari(0.0),
            M_(A.rows()),
            N_(B.cols()),
            A_((double*)stan::agrad::memalloc_.alloc(sizeof(double) 
                                                     * A.rows() * A.cols())),
            C_((double*)stan::agrad::memalloc_.alloc(sizeof(double) 
                                                     * B.rows() * B.cols())),
            _variRefA((vari**)stan::agrad::memalloc_.alloc(sizeof(vari*) 
                                                           * A.rows() 
                                                           * (A.rows() + 1) / 2)),
            _variRefB((vari**)stan::agrad::memalloc_.alloc(sizeof(vari*) 
                                                           * B.rows() * B.cols())),
            _variRefC((vari**)stan::agrad::memalloc_.alloc(sizeof(vari*) 
                                                           * B.rows() * B.cols()))
        {
          using Eigen::Matrix;
          using Eigen::Map;

          size_t pos = 0;
          if (TriView == Eigen::Lower) {
            for (size_type j = 0; j < M_; j++)
              for (size_type i = j; i < M_; i++)
                _variRefA[pos++] = A(i,j).vi_;
          } else if (TriView == Eigen::Upper) {
            for (size_type j = 0; j < M_; j++)
              for (size_type i = 0; i < j+1; i++)
                _variRefA[pos++] = A(i,j).vi_;
          }

          pos = 0;
          for (size_type j = 0; j < M_; j++) {
            for (size_type i = 0; i < M_; i++) {
              A_[pos++] = A(i,j).val();
            }
          }
  
          pos = 0;
          for (size_type j = 0; j < N_; j++) {
            for (size_type i = 0; i < M_; i++) {
              _variRefB[pos] = B(i,j).vi_;
              C_[pos++] = B(i,j).val();
            }
          }
        
          Matrix<double,R1,C2> C(M_,N_);
          C = Map<Matrix<double,R1,C2> >(C_,M_,N_);

          C = Map<Matrix<double,R1,C1> >(A_,M_,M_)
            .template triangularView<TriView>().solve(C);

          pos = 0;
          for (size_type j = 0; j < N_; j++) {
            for (size_type i = 0; i < M_; i++) {
              C_[pos] = C(i,j);
              _variRefC[pos] = new vari(C_[pos],false);
              pos++;
            }
          }

        }
      
        virtual void chain() {
          using Eigen::Matrix;
          using Eigen::Map;
          Matrix<double,R1,C1> adjA(M_,M_);
          Matrix<double,R2,C2> adjB(M_,N_);
          Matrix<double,R1,C2> adjC(M_,N_);

          size_t pos = 0;
          for (size_type j = 0; j < adjC.cols(); j++)
            for (size_type i = 0; i < adjC.rows(); i++)
              adjC(i,j) = _variRefC[pos++]->adj_;
        
          adjB = Map<Matrix<double,R1,C1> >(A_,M_,M_)
            .template triangularView<TriView>().transpose().solve(adjC);
          adjA.noalias() = -adjB
            * Map<Matrix<double,R1,C2> >(C_,M_,N_).transpose();
        
          pos = 0;
          if (TriView == Eigen::Lower) {
            for (size_type j = 0; j < adjA.cols(); j++)
              for (size_type i = j; i < adjA.rows(); i++)
                _variRefA[pos++]->adj_ += adjA(i,j);
          } else if (TriView == Eigen::Upper) {
            for (size_type j = 0; j < adjA.cols(); j++)
              for (size_type i = 0; i < j+1; i++)
                _variRefA[pos++]->adj_ += adjA(i,j);
          } 
        
          pos = 0;
          for (size_type j = 0; j < adjB.cols(); j++)
            for (size_type i = 0; i < adjB.rows(); i++)
              _variRefB[pos++]->adj_ += adjB(i,j);
        }
      };

      template <int TriView,int R1,int C1,int R2,int C2>
      class mdivide_left_tri_dv_vari : public vari {
      public:
        int M_; // A.rows() = A.cols() = B.rows()
        int N_; // B.cols()
        double* A_;
        double* C_;
        vari** _variRefB;
        vari** _variRefC;
      
        mdivide_left_tri_dv_vari(const Eigen::Matrix<double,R1,C1> &A,
                                 const Eigen::Matrix<var,R2,C2> &B)
          : vari(0.0),
            M_(A.rows()),
            N_(B.cols()),
            A_((double*)stan::agrad::memalloc_.alloc(sizeof(double) 
                                                     * A.rows() * A.cols())),
            C_((double*)stan::agrad::memalloc_.alloc(sizeof(double) 
                                                     * B.rows() * B.cols())),
            _variRefB((vari**)stan::agrad::memalloc_.alloc(sizeof(vari*) 
                                                           * B.rows() * B.cols())),
            _variRefC((vari**)stan::agrad::memalloc_.alloc(sizeof(vari*) 
                                                           * B.rows() * B.cols()))
        {
          using Eigen::Matrix;
          using Eigen::Map;

          size_t pos = 0;
          for (size_type j = 0; j < M_; j++) {
            for (size_type i = 0; i < M_; i++) {
              A_[pos++] = A(i,j);
            }
          }

          pos = 0;
          for (size_type j = 0; j < N_; j++) {
            for (size_type i = 0; i < M_; i++) {
              _variRefB[pos] = B(i,j).vi_;
              C_[pos++] = B(i,j).val();
            }
          }

          Matrix<double,R1,C2> C(M_,N_);
          C = Map<Matrix<double,R1,C2> >(C_,M_,N_);
  
          C = Map<Matrix<double,R1,C1> >(A_,M_,M_)
            .template triangularView<TriView>().solve(C);

          pos = 0;
          for (size_type j = 0; j < N_; j++) {
            for (size_type i = 0; i < M_; i++) {
              C_[pos] = C(i,j);
              _variRefC[pos] = new vari(C_[pos],false);
              pos++;
            }
          }
        }
      
        virtual void chain() {
          using Eigen::Matrix;
          using Eigen::Map;
          Matrix<double,R2,C2> adjB(M_,N_);
          Matrix<double,R1,C2> adjC(M_,N_);

          size_t pos = 0;
          for (size_type j = 0; j < adjC.cols(); j++)
            for (size_type i = 0; i < adjC.rows(); i++)
              adjC(i,j) = _variRefC[pos++]->adj_;

          adjB = Map<Matrix<double,R1,C1> >(A_,M_,M_)
            .template triangularView<TriView>().transpose().solve(adjC);
  
          pos = 0;
          for (size_type j = 0; j < adjB.cols(); j++)
            for (size_type i = 0; i < adjB.rows(); i++)
              _variRefB[pos++]->adj_ += adjB(i,j);
        }
      };
    
      template <int TriView,int R1,int C1,int R2,int C2>
      class mdivide_left_tri_vd_vari : public vari {
      public:
        int M_; // A.rows() = A.cols() = B.rows()
        int N_; // B.cols()
        double* A_;
        double* C_;
        vari** _variRefA;
        vari** _variRefC;

        mdivide_left_tri_vd_vari(const Eigen::Matrix<var,R1,C1> &A,
                                 const Eigen::Matrix<double,R2,C2> &B)
          : vari(0.0),
            M_(A.rows()),
            N_(B.cols()),
            A_((double*)stan::agrad::memalloc_.alloc(sizeof(double) 
                                                     * A.rows() * A.cols())),
            C_((double*)stan::agrad::memalloc_.alloc(sizeof(double) 
                                                     * B.rows() * B.cols())),
            _variRefA((vari**)stan::agrad::memalloc_.alloc(sizeof(vari*) 
                                                           * A.rows() 
                                                           * (A.rows() + 1) / 2)),
            _variRefC((vari**)stan::agrad::memalloc_.alloc(sizeof(vari*) 
                                                           * B.rows() * B.cols()))
        {
          using Eigen::Matrix;
          using Eigen::Map;

          size_t pos = 0;
          if (TriView == Eigen::Lower) {
            for (size_type j = 0; j < M_; j++)
              for (size_type i = j; i < M_; i++)
                _variRefA[pos++] = A(i,j).vi_;
          } else if (TriView == Eigen::Upper) {
            for (size_type j = 0; j < M_; j++)
              for (size_type i = 0; i < j+1; i++)
                _variRefA[pos++] = A(i,j).vi_;
          } 

          pos = 0;
          for (size_type j = 0; j < M_; j++) {
            for (size_type i = 0; i < M_; i++) {
              A_[pos++] = A(i,j).val();
            }
          }

          Matrix<double,R1,C2> C(M_,N_);
          C = Map<Matrix<double,R1,C1> >(A_,M_,M_)
            .template triangularView<TriView>().solve(B);

          pos = 0;
          for (size_type j = 0; j < N_; j++) {
            for (size_type i = 0; i < M_; i++) {
              C_[pos] = C(i,j);
              _variRefC[pos] = new vari(C_[pos],false);
              pos++;
            }
          }
        }
      
        virtual void chain() {
          using Eigen::Matrix;
          using Eigen::Map;
          Matrix<double,R1,C1> adjA(M_,M_);
          Matrix<double,R1,C2> adjC(M_,N_);
        
          size_t pos = 0;
          for (size_type j = 0; j < adjC.cols(); j++)
            for (size_type i = 0; i < adjC.rows(); i++)
              adjC(i,j) = _variRefC[pos++]->adj_;

          adjA.noalias() = -Map<Matrix<double,R1,C1> >(A_,M_,M_)
            .template triangularView<TriView>()
            .transpose().solve(adjC*Map<Matrix<double,R1,C2> >(C_,M_,N_).transpose());
  
          pos = 0;
          if (TriView == Eigen::Lower) {
            for (size_type j = 0; j < adjA.cols(); j++)
              for (size_type i = j; i < adjA.rows(); i++)
                _variRefA[pos++]->adj_ += adjA(i,j);
          } else if (TriView == Eigen::Upper) {
            for (size_type j = 0; j < adjA.cols(); j++)
              for (size_type i = 0; i < j+1; i++)
                _variRefA[pos++]->adj_ += adjA(i,j);
          }
        }
      };
    }

    template <int TriView,int R1,int C1,int R2,int C2>
    inline 
    Eigen::Matrix<var,R1,C2>
    mdivide_left_tri(const Eigen::Matrix<var,R1,C1> &A,
                     const Eigen::Matrix<var,R2,C2> &b) {
      Eigen::Matrix<var,R1,C2> res(b.rows(),b.cols());
      
      stan::error_handling::check_square("mdivide_left_tri", "A", A);
      stan::error_handling::check_multiplicable("mdivide_left_tri",
                                                "A", A,
                                                "b", b);
      
      // NOTE: this is not a memory leak, this vari is used in the 
      // expression graph to evaluate the adjoint, but is not needed
      // for the returned matrix.  Memory will be cleaned up with the arena allocator.
      mdivide_left_tri_vv_vari<TriView,R1,C1,R2,C2> *baseVari = new mdivide_left_tri_vv_vari<TriView,R1,C1,R2,C2>(A,b);

      size_t pos = 0;
      for (size_type j = 0; j < res.cols(); j++)
        for (size_type i = 0; i < res.rows(); i++)
          res(i,j).vi_ = baseVari->_variRefC[pos++];
      
      return res;
    }
    template <int TriView,int R1,int C1,int R2,int C2>
    inline 
    Eigen::Matrix<var,R1,C2>
    mdivide_left_tri(const Eigen::Matrix<double,R1,C1> &A,
                     const Eigen::Matrix<var,R2,C2> &b) {
      Eigen::Matrix<var,R1,C2> res(b.rows(),b.cols());
      
      stan::error_handling::check_square("mdivide_left_tri", "A", A);
      stan::error_handling::check_multiplicable("mdivide_left_tri",
                                                "A", A,
                                                "b", b);
      
      // NOTE: this is not a memory leak, this vari is used in the 
      // expression graph to evaluate the adjoint, but is not needed
      // for the returned matrix.  Memory will be cleaned up with the arena allocator.
      mdivide_left_tri_dv_vari<TriView,R1,C1,R2,C2> *baseVari = new mdivide_left_tri_dv_vari<TriView,R1,C1,R2,C2>(A,b);

      size_t pos = 0;
      for (size_type j = 0; j < res.cols(); j++)
        for (size_type i = 0; i < res.rows(); i++)
          res(i,j).vi_ = baseVari->_variRefC[pos++];
      
      return res;
    }
    template <int TriView,int R1,int C1,int R2,int C2>
    inline 
    Eigen::Matrix<var,R1,C2>
    mdivide_left_tri(const Eigen::Matrix<var,R1,C1> &A,
                     const Eigen::Matrix<double,R2,C2> &b) {
      Eigen::Matrix<var,R1,C2> res(b.rows(),b.cols());
      
      stan::error_handling::check_square("mdivide_left_tri", "A", A);
      stan::error_handling::check_multiplicable("mdivide_left_tri",
                                                "A", A,
                                                "b", b);
      
      // NOTE: this is not a memory leak, this vari is used in the 
      // expression graph to evaluate the adjoint, but is not needed
      // for the returned matrix.  Memory will be cleaned up with the arena allocator.
      mdivide_left_tri_vd_vari<TriView,R1,C1,R2,C2> *baseVari = new mdivide_left_tri_vd_vari<TriView,R1,C1,R2,C2>(A,b);
      
      size_t pos = 0;
      for (size_type j = 0; j < res.cols(); j++)
        for (size_type i = 0; i < res.rows(); i++)
          res(i,j).vi_ = baseVari->_variRefC[pos++];
      
      return res;
    }
    
  }
}
#endif
