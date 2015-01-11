#ifndef STAN__AGRAD__REV__MATRIX__TRACE_GEN_QUAD_FORM_HPP
#define STAN__AGRAD__REV__MATRIX__TRACE_GEN_QUAD_FORM_HPP

#include <boost/utility/enable_if.hpp>
#include <boost/type_traits.hpp>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/agrad/rev/matrix/value_of.hpp>
#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/vari.hpp>
#include <stan/agrad/rev/matrix/typedefs.hpp>
#include <stan/math/matrix/trace_gen_quad_form.hpp>
#include <stan/error_handling/matrix/check_multiplicable.hpp>
#include <stan/error_handling/matrix/check_square.hpp>

namespace stan {
  namespace agrad {
    namespace {
      template<typename TD,int RD,int CD,
               typename TA,int RA,int CA,
               typename TB,int RB,int CB>
      class trace_gen_quad_form_vari_alloc : public chainable_alloc {
      public:
        trace_gen_quad_form_vari_alloc(const Eigen::Matrix<TD,RD,CD> &D,
                                       const Eigen::Matrix<TA,RA,CA> &A,
                                       const Eigen::Matrix<TB,RB,CB> &B)
        : D_(D), A_(A), B_(B)
        { }
        
        double compute() {
          return stan::math::trace_gen_quad_form(value_of(D_),
                                                 value_of(A_),
                                                 value_of(B_));
        }
        
        Eigen::Matrix<TD,RD,CD>  D_;
        Eigen::Matrix<TA,RA,CA>  A_;
        Eigen::Matrix<TB,RB,CB>  B_;
      };
      
      template<typename TD,int RD,int CD,
               typename TA,int RA,int CA,
               typename TB,int RB,int CB>
      class trace_gen_quad_form_vari : public vari {
      protected:
        static inline void computeAdjoints(const double &adj,
                                           const Eigen::Matrix<double,RD,CD> &D,
                                           const Eigen::Matrix<double,RA,CA> &A,
                                           const Eigen::Matrix<double,RB,CB> &B,
                                           Eigen::Matrix<var,RD,CD> *varD,
                                           Eigen::Matrix<var,RA,CA> *varA,
                                           Eigen::Matrix<var,RB,CB> *varB)
        {
          Eigen::Matrix<double,CA,CB> AtB;
          Eigen::Matrix<double,RA,CB> BD;
          if (varB || varA)
            BD.noalias() = B*D;
          if (varB || varD)
            AtB.noalias() = A.transpose()*B;
          
          if (varB) {
            Eigen::Matrix<double,RB,CB> adjB(adj*(A*BD + AtB*D.transpose()));
            int i,j;
            for (j = 0; j < B.cols(); j++)
              for (i = 0; i < B.rows(); i++)
                (*varB)(i,j).vi_->adj_ += adjB(i,j);
          }
          if (varA) {
            Eigen::Matrix<double,RA,CA> adjA(adj*(B*BD.transpose()));
            int i,j;
            for (j = 0; j < A.cols(); j++)
              for (i = 0; i < A.rows(); i++)
                (*varA)(i,j).vi_->adj_ += adjA(i,j);
          }
          if (varD) {
            Eigen::Matrix<double,RD,CD> adjD(adj*(B.transpose()*AtB));
            int i,j;
            for (j = 0; j < D.cols(); j++)
              for (i = 0; i < D.rows(); i++)
                (*varD)(i,j).vi_->adj_ += adjD(i,j);
          }
        }

        
      public:
        trace_gen_quad_form_vari(trace_gen_quad_form_vari_alloc<TD,RD,CD,TA,RA,CA,TB,RB,CB> *impl)
        : vari(impl->compute()), _impl(impl) { }
        
        virtual void chain() {
          computeAdjoints(adj_,
                          value_of(_impl->D_),
                          value_of(_impl->A_),
                          value_of(_impl->B_),
                          (Eigen::Matrix<var,RD,CD>*)(boost::is_same<TD,var>::value?(&_impl->D_):NULL),
                          (Eigen::Matrix<var,RA,CA>*)(boost::is_same<TA,var>::value?(&_impl->A_):NULL),
                          (Eigen::Matrix<var,RB,CB>*)(boost::is_same<TB,var>::value?(&_impl->B_):NULL));
        }
        
        trace_gen_quad_form_vari_alloc<TD,RD,CD,TA,RA,CA,TB,RB,CB> *_impl;
      };
    }
    
    template<typename TD,int RD,int CD,
             typename TA,int RA,int CA,
             typename TB,int RB,int CB>
    inline typename
    boost::enable_if_c< boost::is_same<TD,var>::value ||
                        boost::is_same<TA,var>::value ||
                        boost::is_same<TB,var>::value,
                        var >::type
    trace_gen_quad_form(const Eigen::Matrix<TD,RD,CD> &D,
                        const Eigen::Matrix<TA,RA,CA> &A,
                        const Eigen::Matrix<TB,RB,CB> &B)
    {
      stan::error_handling::check_square("trace_gen_quad_form", "A", A);
      stan::error_handling::check_square("trace_gen_quad_form", "D", D);
      stan::error_handling::check_multiplicable("trace_gen_quad_form", 
                                                "A", A, 
                                                "B", B);
      stan::error_handling::check_multiplicable("trace_gen_quad_form",
                                                "B", B,
                                                "D", D);
      
      trace_gen_quad_form_vari_alloc<TD,RD,CD,TA,RA,CA,TB,RB,CB> *baseVari = new trace_gen_quad_form_vari_alloc<TD,RD,CD,TA,RA,CA,TB,RB,CB>(D,A,B);
      
      return var(new trace_gen_quad_form_vari<TD,RD,CD,TA,RA,CA,TB,RB,CB>(baseVari));
    }
  }
}

#endif
