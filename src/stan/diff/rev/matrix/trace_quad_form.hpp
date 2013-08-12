#ifndef __STAN__AGRAD__REV__MATRIX__TRACE_QUAD_FORM_HPP__
#define __STAN__AGRAD__REV__MATRIX__TRACE_QUAD_FORM_HPP__

#include <boost/utility/enable_if.hpp>
#include <boost/type_traits.hpp>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/agrad/rev/matrix/value_of.hpp>
#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/vari.hpp>
#include <stan/agrad/rev/matrix/typedefs.hpp>
#include <stan/math/matrix/trace_quad_form.hpp>
#include <stan/math/matrix/validate_multiplicable.hpp>
#include <stan/math/matrix/validate_square.hpp>

namespace stan {
  namespace agrad {
    namespace {
      template<typename TA,int RA,int CA,typename TB,int RB,int CB>
      class trace_quad_form_vari_alloc : public chainable_alloc {
      public:
        trace_quad_form_vari_alloc(const Eigen::Matrix<TA,RA,CA> &A,
                                   const Eigen::Matrix<TB,RB,CB> &B)
        : _A(A), _B(B)
        { }
        
        double compute() {
          return stan::math::trace_quad_form(value_of(_A),
                                             value_of(_B));
        }
        
        Eigen::Matrix<TA,RA,CA>  _A;
        Eigen::Matrix<TB,RB,CB>  _B;
      };
      
      template<typename TA,int RA,int CA,typename TB,int RB,int CB>
      class trace_quad_form_vari : public vari {
      protected:
        static inline void chainA(Eigen::Matrix<double,RA,CA> &A, 
                                  const Eigen::Matrix<double,RB,CB> &Bd,
                                  const double &adjC) {}
        static inline void chainB(Eigen::Matrix<double,RB,CB> &B, 
                                  const Eigen::Matrix<double,RA,CA> &Ad,
                                  const Eigen::Matrix<double,RB,CB> &Bd,
                                  const double &adjC) {}
        
        static inline void chainA(Eigen::Matrix<var,RA,CA> &A, 
                                  const Eigen::Matrix<double,RB,CB> &Bd,
                                  const double &adjC)
        {
          size_t i,j;
          Eigen::Matrix<double,RA,CA>     adjA(adjC*Bd*Bd.transpose());
          for (j = 0; j < A.cols(); j++)
            for (i = 0; i < A.rows(); i++)
              A(i,j).vi_->adj_ += adjA(i,j);
        }
        static inline void chainB(Eigen::Matrix<var,RB,CB> &B, 
                                  const Eigen::Matrix<double,RA,CA> &Ad,
                                  const Eigen::Matrix<double,RB,CB> &Bd,
                                  const double &adjC)
        {
          size_t i,j;
          Eigen::Matrix<double,RA,CA>     adjB(adjC*(Ad + Ad.transpose())*Bd);
          for (j = 0; j < B.cols(); j++)
            for (i = 0; i < B.rows(); i++)
              B(i,j).vi_->adj_ += adjB(i,j);
        }
        
        inline void chainAB(Eigen::Matrix<TA,RA,CA> &A,
                            Eigen::Matrix<TB,RB,CB> &B,
                            const Eigen::Matrix<double,RA,CA> &Ad,
                            const Eigen::Matrix<double,RB,CB> &Bd,
                            const double &adjC)
        {
          chainA(A,Bd,adjC);
          chainB(B,Ad,Bd,adjC);
        }
        

      public:
        trace_quad_form_vari(trace_quad_form_vari_alloc<TA,RA,CA,TB,RB,CB> *impl)
        : vari(impl->compute()), _impl(impl) { }
        
        virtual void chain() {
          chainAB(_impl->_A, _impl->_B,
                  value_of(_impl->_A), value_of(_impl->_B),
                  adj_);
        };

        trace_quad_form_vari_alloc<TA,RA,CA,TB,RB,CB> *_impl;
      };
    }
    
    template<typename TA,int RA,int CA,typename TB,int RB,int CB>
    inline typename
    boost::enable_if_c< boost::is_same<TA,var>::value ||
                        boost::is_same<TB,var>::value,
                        var >::type
    trace_quad_form(const Eigen::Matrix<TA,RA,CA> &A,
                    const Eigen::Matrix<TB,RB,CB> &B)
    {
      stan::math::validate_square(A,"trace_quad_form");
      stan::math::validate_multiplicable(A,B,"trace_quad_form");
      
      trace_quad_form_vari_alloc<TA,RA,CA,TB,RB,CB> *baseVari = new trace_quad_form_vari_alloc<TA,RA,CA,TB,RB,CB>(A,B);
      
      return var(new trace_quad_form_vari<TA,RA,CA,TB,RB,CB>(baseVari));
    }
  }
}

#endif
