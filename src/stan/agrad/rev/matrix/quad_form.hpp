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
          size_t i,j;
          Eigen::Matrix<double,CB,CB> Cd(B.transpose()*A*B);
          for (j = 0; j < _C.cols(); j++) {
            for (i = 0; i < _C.rows(); i++) {
              if (_sym) {
                _C(i,j) = var(new vari(0.5*(Cd(i,j) + Cd(j,i)),false));
              }
              else {
                _C(i,j) = var(new vari(Cd(i,j),false));
              }
            }
          }
        }
                              
      public:
        quad_form_vari_alloc(const Eigen::Matrix<TA,RA,CA> &A,
                             const Eigen::Matrix<TB,RB,CB> &B,
                             bool symmetric = false)
        : _A(A), _B(B), _C(_B.cols(),_B.cols()), _sym(symmetric)
        {
          compute(value_of(A),value_of(B));
        }
        
        Eigen::Matrix<TA,RA,CA>  _A;
        Eigen::Matrix<TB,RB,CB>  _B;
        Eigen::Matrix<var,CB,CB> _C;
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
          size_t i,j;
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
          size_t i,j;
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
          size_t i,j;
          Eigen::Matrix<double,CB,CB> adjC(_impl->_C.rows(),_impl->_C.cols());
          
          for (j = 0; j < _impl->_C.cols(); j++)
            for (i = 0; i < _impl->_C.rows(); i++)
              adjC(i,j) = _impl->_C(i,j).vi_->adj_;
          
          chainAB(_impl->_A, _impl->_B,
                  value_of(_impl->_A), value_of(_impl->_B),
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
      
      return baseVari->_impl->_C;
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
      
      return baseVari->_impl->_C(0,0);
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
      
      return baseVari->_impl->_C;
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
      
      return baseVari->_impl->_C(0,0);
    }
  }
}

#endif
