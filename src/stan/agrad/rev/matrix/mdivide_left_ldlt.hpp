#ifndef STAN__AGRAD__REV__MATRIX__MDIVIDE_LEFT_LDLT_HPP
#define STAN__AGRAD__REV__MATRIX__MDIVIDE_LEFT_LDLT_HPP

#include <stan/math/matrix/Eigen.hpp>
#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/matrix/LDLT_alloc.hpp>
#include <stan/agrad/rev/matrix/LDLT_factor.hpp>
#include <stan/error_handling/matrix/check_multiplicable.hpp>

namespace stan {
  namespace agrad {
    namespace {
      template <int R1,int C1,int R2,int C2>
      class mdivide_left_ldlt_alloc : public chainable_alloc {
      public:
        virtual ~mdivide_left_ldlt_alloc() {}
        
        /**
         * This share_ptr is used to prevent copying the LDLT factorizations
         * for mdivide_left_ldlt(ldltA,b) when ldltA is a LDLT_factor<double>.
         * The pointer is shared with the LDLT_factor<double> class.
         **/
        boost::shared_ptr< Eigen::LDLT< Eigen::Matrix<double,R1,C1> > > _ldltP;
        Eigen::Matrix<double,R2,C2> C_;
      };
      
      /**
       * The vari for mdivide_left_ldlt(A,b) which handles the chain() call
       * for all elements of the result.  This vari follows the pattern
       * used in the other matrix operations where there is one "master"
       * vari whose value is never used and a large number of "slave" varis
       * whose chain() functions are never called because their adjoints are
       * set by the "mater" vari.
       *
       * This class handles the var/var case.
       **/
      template <int R1,int C1,int R2,int C2>
      class mdivide_left_ldlt_vv_vari : public vari {
      public:
        int M_; // A.rows() = A.cols() = B.rows()
        int N_; // B.cols()
        vari** _variRefB;
        vari** _variRefC;
        mdivide_left_ldlt_alloc<R1,C1,R2,C2> *_alloc;
        const LDLT_alloc<R1,C1> *_alloc_ldlt;
        
        mdivide_left_ldlt_vv_vari(const stan::math::LDLT_factor<var,R1,C1> &A,
                                  const Eigen::Matrix<var,R2,C2> &B)
          : vari(0.0),
            M_(A.rows()),
            N_(B.cols()),
            _variRefB((vari**)stan::agrad::memalloc_.alloc(sizeof(vari*) 
                                                           * B.rows() * B.cols())),
            _variRefC((vari**)stan::agrad::memalloc_.alloc(sizeof(vari*) 
                                                           * B.rows() * B.cols())),
          _alloc(new mdivide_left_ldlt_alloc<R1,C1,R2,C2>()),
          _alloc_ldlt(A._alloc)
        {
          int pos = 0;
          _alloc->C_.resize(M_,N_);
          for (int j = 0; j < N_; j++) {
            for (int i = 0; i < M_; i++) {
              _variRefB[pos] = B(i,j).vi_;
              _alloc->C_(i,j) = B(i,j).val();
              pos++;
            }
          }
          
          _alloc_ldlt->_ldlt.solveInPlace(_alloc->C_);
          
          pos = 0;
          for (int j = 0; j < N_; j++) {
            for (int i = 0; i < M_; i++) {
              _variRefC[pos] = new vari(_alloc->C_(i,j),false);
              pos++;
            }
          }
        }
        
        virtual void chain() {
          Eigen::Matrix<double,R1,C1> adjA(M_,M_);
          Eigen::Matrix<double,R2,C2> adjB(M_,N_);
          
          int pos = 0;
          for (int j = 0; j < N_; j++)
            for (int i = 0; i < M_; i++)
              adjB(i,j) = _variRefC[pos++]->adj_;
          
          _alloc_ldlt->_ldlt.solveInPlace(adjB);
          adjA.noalias() = -adjB * _alloc->C_.transpose();

          for (int j = 0; j < M_; j++)
            for (int i = 0; i < M_; i++)
              _alloc_ldlt->_variA(i,j)->adj_ += adjA(i,j);
          
          pos = 0;
          for (int j = 0; j < N_; j++)
            for (int i = 0; i < M_; i++)
              _variRefB[pos++]->adj_ += adjB(i,j);
        }
      };

      /**
       * The vari for mdivide_left_ldlt(A,b) which handles the chain() call
       * for all elements of the result.  This vari follows the pattern
       * used in the other matrix operations where there is one "master"
       * vari whose value is never used and a large number of "slave" varis
       * whose chain() functions are never called because their adjoints are
       * set by the "mater" vari.
       *
       * This class handles the double/var case.
       **/
      template <int R1,int C1,int R2,int C2>
      class mdivide_left_ldlt_dv_vari : public vari {
      public:
        int M_; // A.rows() = A.cols() = B.rows()
        int N_; // B.cols()
        vari** _variRefB;
        vari** _variRefC;
        mdivide_left_ldlt_alloc<R1,C1,R2,C2> *_alloc;
        
        mdivide_left_ldlt_dv_vari(const stan::math::LDLT_factor<double,R1,C1> &A,
                                  const Eigen::Matrix<var,R2,C2> &B)
          : vari(0.0),
            M_(A.rows()),
            N_(B.cols()),
            _variRefB((vari**)stan::agrad::memalloc_.alloc(sizeof(vari*) 
                                                           * B.rows() * B.cols())),
            _variRefC((vari**)stan::agrad::memalloc_.alloc(sizeof(vari*) 
                                                           * B.rows() * B.cols())),
          _alloc(new mdivide_left_ldlt_alloc<R1,C1,R2,C2>())
        {
          using Eigen::Matrix;
          using Eigen::Map;
          
          int pos = 0;
          _alloc->C_.resize(M_,N_);
          for (int j = 0; j < N_; j++) {
            for (int i = 0; i < M_; i++) {
              _variRefB[pos] = B(i,j).vi_;
              _alloc->C_(i,j) = B(i,j).val();
              pos++;
            }
          }
          
          _alloc->_ldltP = A._ldltP;
          _alloc->_ldltP->solveInPlace(_alloc->C_);
          
          pos = 0;
          for (int j = 0; j < N_; j++) {
            for (int i = 0; i < M_; i++) {
              _variRefC[pos] = new vari(_alloc->C_(i,j),false);
              pos++;
            }
          }
        }
        
        virtual void chain() {
          Eigen::Matrix<double,R2,C2> adjB(M_,N_);
          
          int pos = 0;
          for (int j = 0; j < adjB.cols(); j++)
            for (int i = 0; i < adjB.rows(); i++)
              adjB(i,j) = _variRefC[pos++]->adj_;
          
          _alloc->_ldltP->solveInPlace(adjB);
          
          pos = 0;
          for (int j = 0; j < adjB.cols(); j++)
            for (int i = 0; i < adjB.rows(); i++)
              _variRefB[pos++]->adj_ += adjB(i,j);
        }
      };

      /**
       * The vari for mdivide_left_ldlt(A,b) which handles the chain() call
       * for all elements of the result.  This vari follows the pattern
       * used in the other matrix operations where there is one "master"
       * vari whose value is never used and a large number of "slave" varis
       * whose chain() functions are never called because their adjoints are
       * set by the "mater" vari.
       *
       * This class handles the var/double case.
       **/
      template <int R1,int C1,int R2,int C2>
      class mdivide_left_ldlt_vd_vari : public vari {
      public:
        int M_; // A.rows() = A.cols() = B.rows()
        int N_; // B.cols()
        vari** _variRefC;
        mdivide_left_ldlt_alloc<R1,C1,R2,C2> *_alloc;
        const LDLT_alloc<R1,C1> *_alloc_ldlt;
      
        mdivide_left_ldlt_vd_vari(const stan::math::LDLT_factor<var,R1,C1> &A,
                                  const Eigen::Matrix<double,R2,C2> &B)
          : vari(0.0),
            M_(A.rows()),
            N_(B.cols()),
            _variRefC((vari**)stan::agrad::memalloc_.alloc(sizeof(vari*) 
                                                           * B.rows() * B.cols())),
            _alloc(new mdivide_left_ldlt_alloc<R1,C1,R2,C2>()),
            _alloc_ldlt(A._alloc)
        {
          _alloc->C_ = B;
          _alloc_ldlt->_ldlt.solveInPlace(_alloc->C_);
          
          int pos = 0;
          for (int j = 0; j < N_; j++) {
            for (int i = 0; i < M_; i++) {
              _variRefC[pos] = new vari(_alloc->C_(i,j),false);
              pos++;
            }
          }
        }
      
        virtual void chain() {
          Eigen::Matrix<double,R1,C1> adjA(M_,M_);
          Eigen::Matrix<double,R1,C2> adjC(M_,N_);

          int pos = 0;
          for (int j = 0; j < adjC.cols(); j++)
            for (int i = 0; i < adjC.rows(); i++)
              adjC(i,j) = _variRefC[pos++]->adj_;
        
          adjA = -_alloc_ldlt->_ldlt.solve(adjC*_alloc->C_.transpose());

          for (int j = 0; j < adjA.cols(); j++)
            for (int i = 0; i < adjA.rows(); i++)
              _alloc_ldlt->_variA(i,j)->adj_ += adjA(i,j);
        }
      };
    }

    /**
     * Returns the solution of the system Ax=b given an LDLT_factor of A
     * @param A LDLT_factor
     * @param b Right hand side matrix or vector.
     * @return x = b A^-1, solution of the linear system.
     * @throws std::domain_error if rows of b don't match the size of A.
     */
    template <int R1,int C1,int R2,int C2>
    inline Eigen::Matrix<var,R1,C2>
    mdivide_left_ldlt(const stan::math::LDLT_factor<var,R1,C1> &A,
                      const Eigen::Matrix<var,R2,C2> &b) {
      Eigen::Matrix<var,R1,C2> res(b.rows(),b.cols());

      stan::error_handling::check_multiplicable("mdivide_left_ldlt",
                                                "A", A,
                                                "b", b);
     
      mdivide_left_ldlt_vv_vari<R1,C1,R2,C2> *baseVari = new mdivide_left_ldlt_vv_vari<R1,C1,R2,C2>(A,b);
      
      int pos = 0;
      for (int j = 0; j < res.cols(); j++)
        for (int i = 0; i < res.rows(); i++)
          res(i,j).vi_ = baseVari->_variRefC[pos++];
      
      return res;
    }

    /**
     * Returns the solution of the system Ax=b given an LDLT_factor of A
     * @param A LDLT_factor
     * @param b Right hand side matrix or vector.
     * @return x = b A^-1, solution of the linear system.
     * @throws std::domain_error if rows of b don't match the size of A.
     */
    template <int R1,int C1,int R2,int C2>
    inline Eigen::Matrix<var,R1,C2>
    mdivide_left_ldlt(const stan::math::LDLT_factor<var,R1,C1> &A,
                      const Eigen::Matrix<double,R2,C2> &b) {
      Eigen::Matrix<var,R1,C2> res(b.rows(),b.cols());
      
      stan::error_handling::check_multiplicable("mdivide_left_ldlt",
                                                "A", A,
                                                "b", b);

      mdivide_left_ldlt_vd_vari<R1,C1,R2,C2> *baseVari = new mdivide_left_ldlt_vd_vari<R1,C1,R2,C2>(A,b);
      
      int pos = 0;
      for (int j = 0; j < res.cols(); j++)
        for (int i = 0; i < res.rows(); i++)
          res(i,j).vi_ = baseVari->_variRefC[pos++];
      
      return res;
    }
    
    /**
     * Returns the solution of the system Ax=b given an LDLT_factor of A
     * @param A LDLT_factor
     * @param b Right hand side matrix or vector.
     * @return x = b A^-1, solution of the linear system.
     * @throws std::domain_error if rows of b don't match the size of A.
     */
    template <int R1,int C1,int R2,int C2>
    inline Eigen::Matrix<var,R1,C2>
    mdivide_left_ldlt(const stan::math::LDLT_factor<double,R1,C1> &A,
                      const Eigen::Matrix<var,R2,C2> &b) {
      Eigen::Matrix<var,R1,C2> res(b.rows(),b.cols());
      
      stan::error_handling::check_multiplicable("mdivide_left_ldlt",
                                                "A", A,
                                                "b", b);
      
      mdivide_left_ldlt_dv_vari<R1,C1,R2,C2> *baseVari = new mdivide_left_ldlt_dv_vari<R1,C1,R2,C2>(A,b);
      
      int pos = 0;
      for (int j = 0; j < res.cols(); j++)
        for (int i = 0; i < res.rows(); i++)
          res(i,j).vi_ = baseVari->_variRefC[pos++];
      
      return res;
    }

  }
}
#endif
