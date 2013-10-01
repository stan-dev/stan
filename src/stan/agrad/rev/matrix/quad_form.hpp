#ifndef __STAN__AGRAD__REV__MATRIX__QUAD_FORM_HPP__
#define __STAN__AGRAD__REV__MATRIX__QUAD_FORM_HPP__

#include <boost/utility/enable_if.hpp>
#include <boost/type_traits.hpp>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/agrad/var.hpp>
#include <stan/agrad/rev/matrix/typedefs.hpp>
#include <stan/agrad/rev/matrix/value_of.hpp>
#include <stan/math/matrix/quad_form.hpp>
#include <stan/math/matrix/validate_multiplicable.hpp>
#include <stan/math/matrix/validate_square.hpp>
#include <stan/math/matrix/validate_symmetric.hpp>

namespace stan {
  namespace agrad {
    namespace {
      template<typename TA,int RA,int CA,typename TB,int RB,int CB>
      class quad_form_vari_alloc : public chainable_alloc {
      private:
        inline void compute(const Eigen::Matrix<double,RA,CA> &A,
                            const Eigen::Matrix<double,RB,CB> &B)
        {
          int i,j;
          Eigen::Matrix<double,CB,CB> Cd(B.transpose()*A*B);
          for (j = 0; j < C_.cols(); j++) {
            for (i = 0; i < C_.rows(); i++) {
              if (_sym) {
                C_(i,j) = var(new vari(0.5*(Cd(i,j) + Cd(j,i)),false));
              }
              else {
                C_(i,j) = var(new vari(Cd(i,j),false));
              }
            }
          }
        }
                              
      public:
        quad_form_vari_alloc(const Eigen::Matrix<TA,RA,CA> &A,
                             const Eigen::Matrix<TB,RB,CB> &B,
                             bool symmetric = false)
        : A_(A), B_(B), C_(B_.cols(),B_.cols()), _sym(symmetric)
        {
          compute(value_of(A),value_of(B));
        }
        
        Eigen::Matrix<TA,RA,CA>  A_;
        Eigen::Matrix<TB,RB,CB>  B_;
        Eigen::Matrix<var,CB,CB> C_;
        bool                     _sym;
      };
      
      template<typename TA,int RA,int CA,typename TB,int RB,int CB>
      class quad_form_vari : public vari {
      protected:
        inline void chainA(Eigen::Matrix<double,RA,CA> &A, 
                           const Eigen::Matrix<double,RB,CB> &Bd,
                           const Eigen::Matrix<double,CB,CB> &adjC) {}
        inline void chainB(Eigen::Matrix<double,RB,CB> &B,
                           const Eigen::Matrix<double,RA,CA> &Ad,
                           const Eigen::Matrix<double,RB,CB> &Bd,
                           const Eigen::Matrix<double,CB,CB> &adjC) {}
        
        inline void chainA(Eigen::Matrix<var,RA,CA> &A,
                           const Eigen::Matrix<double,RB,CB> &Bd,
                           const Eigen::Matrix<double,CB,CB> &adjC)
        {
          int i,j;
          Eigen::Matrix<double,RA,CA>     adjA(Bd*adjC*Bd.transpose());
          for (j = 0; j < A.cols(); j++) {
            for (i = 0; i < A.rows(); i++) {
              A(i,j).vi_->adj_ += adjA(i,j);
            }
          }
        }
        inline void chainB(Eigen::Matrix<var,RB,CB> &B,
                           const Eigen::Matrix<double,RA,CA> &Ad,
                           const Eigen::Matrix<double,RB,CB> &Bd,
                           const Eigen::Matrix<double,CB,CB> &adjC)
        {
          int i,j;
          Eigen::Matrix<double,RA,CA>     adjB(Ad*Bd*adjC.transpose() + Ad.transpose()*Bd*adjC);
          for (j = 0; j < B.cols(); j++)
            for (i = 0; i < B.rows(); i++)
              B(i,j).vi_->adj_ += adjB(i,j);
        }
        
        inline void chainAB(Eigen::Matrix<TA,RA,CA> &A,
                            Eigen::Matrix<TB,RB,CB> &B,
                            const Eigen::Matrix<double,RA,CA> &Ad,
                            const Eigen::Matrix<double,RB,CB> &Bd,
                            const Eigen::Matrix<double,CB,CB> &adjC)
        {
          chainA(A,Bd,adjC);
          chainB(B,Ad,Bd,adjC);
        }

      public:
        quad_form_vari(const Eigen::Matrix<TA,RA,CA> &A,
                       const Eigen::Matrix<TB,RB,CB> &B,
                       bool symmetric = false)
        : vari(0.0) {
          _impl = new quad_form_vari_alloc<TA,RA,CA,TB,RB,CB>(A,B,symmetric);
        }
        
        virtual void chain() {
          int i,j;
          Eigen::Matrix<double,CB,CB> adjC(_impl->C_.rows(),_impl->C_.cols());
          
          for (j = 0; j < _impl->C_.cols(); j++)
            for (i = 0; i < _impl->C_.rows(); i++)
              adjC(i,j) = _impl->C_(i,j).vi_->adj_;
          
          chainAB(_impl->A_, _impl->B_,
                  value_of(_impl->A_), value_of(_impl->B_),
                  adjC);
        };

        quad_form_vari_alloc<TA,RA,CA,TB,RB,CB> *_impl;
      };
    }
    
    template<typename TA,int RA,int CA,typename TB,int RB,int CB>
    inline typename
    boost::enable_if_c< boost::is_same<TA,var>::value ||
                        boost::is_same<TB,var>::value,
                        Eigen::Matrix<var,CB,CB> >::type
    quad_form(const Eigen::Matrix<TA,RA,CA> &A,
              const Eigen::Matrix<TB,RB,CB> &B)
    {
      stan::math::validate_square(A,"quad_form");
      stan::math::validate_multiplicable(A,B,"quad_form");
      
      quad_form_vari<TA,RA,CA,TB,RB,CB> *baseVari = new quad_form_vari<TA,RA,CA,TB,RB,CB>(A,B);
      
      return baseVari->_impl->C_;
    }
    template<typename TA,int RA,int CA,typename TB,int RB>
    inline typename
    boost::enable_if_c< boost::is_same<TA,var>::value ||
                        boost::is_same<TB,var>::value,
                        var >::type
    quad_form(const Eigen::Matrix<TA,RA,CA> &A,
              const Eigen::Matrix<TB,RB,1> &B)
    {
      stan::math::validate_square(A,"quad_form");
      stan::math::validate_multiplicable(A,B,"quad_form");
      
      quad_form_vari<TA,RA,CA,TB,RB,1> *baseVari = new quad_form_vari<TA,RA,CA,TB,RB,1>(A,B);
      
      return baseVari->_impl->C_(0,0);
    }
    
    template<typename TA,int RA,int CA,typename TB,int RB,int CB>
    inline typename
    boost::enable_if_c< boost::is_same<TA,var>::value ||
                        boost::is_same<TB,var>::value,
                        Eigen::Matrix<var,CB,CB> >::type
    quad_form_sym(const Eigen::Matrix<TA,RA,CA> &A,
                  const Eigen::Matrix<TB,RB,CB> &B)
    {
      stan::math::validate_square(A,"quad_form_sym");
      stan::math::validate_symmetric(A,"quad_form_sym");
      stan::math::validate_multiplicable(A,B,"quad_form_sym");
      
      quad_form_vari<TA,RA,CA,TB,RB,CB> *baseVari = new quad_form_vari<TA,RA,CA,TB,RB,CB>(A,B,true);
      
      return baseVari->_impl->C_;
    }
    template<typename TA,int RA,int CA,typename TB,int RB>
    inline typename
    boost::enable_if_c< boost::is_same<TA,var>::value ||
                        boost::is_same<TB,var>::value,
                        var >::type
    quad_form_sym(const Eigen::Matrix<TA,RA,CA> &A,
                  const Eigen::Matrix<TB,RB,1> &B)
    {
      stan::math::validate_square(A,"quad_form_sym");
      stan::math::validate_symmetric(A,"quad_form_sym");
      stan::math::validate_multiplicable(A,B,"quad_form_sym");
      
      quad_form_vari<TA,RA,CA,TB,RB,1> *baseVari = new quad_form_vari<TA,RA,CA,TB,RB,1>(A,B,true);
      
      return baseVari->_impl->C_(0,0);
    }
  }
}

#endif
