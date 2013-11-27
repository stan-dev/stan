#ifndef __STAN__AGRAD__REV__MATRIX__MDIVIDE_LEFT_HPP__
#define __STAN__AGRAD__REV__MATRIX__MDIVIDE_LEFT_HPP__

#include <stan/math/matrix/Eigen.hpp>
#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/matrix/LDLT_alloc.hpp>
#include <stan/agrad/rev/matrix/LDLT_factor.hpp>
#include <stan/math/matrix/validate_multiplicable.hpp>

namespace stan {
  namespace agrad {
    namespace {
      template <int R1,int C1,int R2,int C2>
      class mdivide_left_ldlt_alloc : public chainable_alloc {
      public:
        virtual ~mdivide_left_ldlt_alloc() {}
        
        boost::shared_ptr< Eigen::LDLT< Eigen::Matrix<double,R1,C1> > > _ldltP;
        Eigen::Matrix<double,R2,C2> C_;
      };
      
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
          size_t pos = 0;
          _alloc->C_.resize(M_,N_);
          for (size_t j = 0; j < N_; j++) {
            for (size_t i = 0; i < M_; i++) {
              _variRefB[pos] = B(i,j).vi_;
              _alloc->C_(i,j) = B(i,j).val();
              pos++;
            }
          }
          
          _alloc_ldlt->_ldlt.solveInPlace(_alloc->C_);
          
          pos = 0;
          for (size_t j = 0; j < N_; j++) {
            for (size_t i = 0; i < M_; i++) {
              _variRefC[pos] = new vari(_alloc->C_(i,j),false);
              pos++;
            }
          }
        }
        
        virtual void chain() {
          Eigen::Matrix<double,R1,C1> adjA(M_,M_);
          Eigen::Matrix<double,R2,C2> adjB(M_,N_);
          
          size_t pos = 0;
          for (size_t j = 0; j < N_; j++)
            for (size_t i = 0; i < M_; i++)
              adjB(i,j) = _variRefC[pos++]->adj_;
          
          _alloc_ldlt->_ldlt.solveInPlace(adjB);
          adjA.noalias() = -adjB * _alloc->C_.transpose();

          for (size_t j = 0; j < M_; j++)
            for (size_t i = 0; i < M_; i++)
              _alloc_ldlt->_variA(i,j)->adj_ += adjA(i,j);
          
          pos = 0;
          for (size_t j = 0; j < N_; j++)
            for (size_t i = 0; i < M_; i++)
              _variRefB[pos++]->adj_ += adjB(i,j);
        }
      };

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
          
          size_t pos = 0;
          _alloc->C_.resize(M_,N_);
          for (size_t j = 0; j < N_; j++) {
            for (size_t i = 0; i < M_; i++) {
              _variRefB[pos] = B(i,j).vi_;
              _alloc->C_(i,j) = B(i,j).val();
              pos++;
            }
          }
          
          _alloc->_ldltP = A._ldltP;
          _alloc->_ldltP->solveInPlace(_alloc->C_);
          
          pos = 0;
          for (size_t j = 0; j < N_; j++) {
            for (size_t i = 0; i < M_; i++) {
              _variRefC[pos] = new vari(_alloc->C_(i,j),false);
              pos++;
            }
          }
        }
        
        virtual void chain() {
          Eigen::Matrix<double,R2,C2> adjB(M_,N_);
          
          size_t pos = 0;
          for (size_t j = 0; j < adjB.cols(); j++)
            for (size_t i = 0; i < adjB.rows(); i++)
              adjB(i,j) = _variRefC[pos++]->adj_;
          
          _alloc->_ldltP->solveInPlace(adjB);
          
          pos = 0;
          for (size_t j = 0; j < adjB.cols(); j++)
            for (size_t i = 0; i < adjB.rows(); i++)
              _variRefB[pos++]->adj_ += adjB(i,j);
        }
      };

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
          
          size_t pos = 0;
          for (size_t j = 0; j < N_; j++) {
            for (size_t i = 0; i < M_; i++) {
              _variRefC[pos] = new vari(_alloc->C_(i,j),false);
              pos++;
            }
          }
        }
      
        virtual void chain() {
          Eigen::Matrix<double,R1,C1> adjA(M_,M_);
          Eigen::Matrix<double,R1,C2> adjC(M_,N_);

          size_t pos = 0;
          for (size_t j = 0; j < adjC.cols(); j++)
            for (size_t i = 0; i < adjC.rows(); i++)
              adjC(i,j) = _variRefC[pos++]->adj_;
        
          adjA = -_alloc_ldlt->_ldlt.solve(adjC*_alloc->C_.transpose());

          for (size_t j = 0; j < adjA.cols(); j++)
            for (size_t i = 0; i < adjA.rows(); i++)
              _alloc_ldlt->_variA(i,j)->adj_ += adjA(i,j);
        }
      };
    }

    
    template <int R1,int C1,int R2,int C2>
    inline Eigen::Matrix<var,R1,C2>
    mdivide_left_ldlt(const stan::math::LDLT_factor<var,R1,C1> &A,
                      const Eigen::Matrix<var,R2,C2> &b) {
      Eigen::Matrix<var,R1,C2> res(b.rows(),b.cols());

      stan::math::validate_multiplicable(A,b,"mdivide_left_ldlt");
      
      mdivide_left_ldlt_vv_vari<R1,C1,R2,C2> *baseVari = new mdivide_left_ldlt_vv_vari<R1,C1,R2,C2>(A,b);
      
      size_t pos = 0;
      for (size_t j = 0; j < res.cols(); j++)
        for (size_t i = 0; i < res.rows(); i++)
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
      for (size_t j = 0; j < res.cols(); j++)
        for (size_t i = 0; i < res.rows(); i++)
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
      for (size_t j = 0; j < res.cols(); j++)
        for (size_t i = 0; i < res.rows(); i++)
          res(i,j).vi_ = baseVari->_variRefC[pos++];
      
      return res;
    }

  }
}
#endif
