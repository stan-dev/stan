#ifndef STAN_MATH_REV_MAT_FUN_TRACE_GEN_QUAD_FORM_HPP
#define STAN_MATH_REV_MAT_FUN_TRACE_GEN_QUAD_FORM_HPP

#include <boost/utility/enable_if.hpp>
#include <boost/type_traits.hpp>
#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/rev/scal/fun/value_of.hpp>
#include <stan/math/prim/mat/fun/value_of.hpp>
#include <stan/math/rev/mat/fun/typedefs.hpp>
#include <stan/math/prim/mat/fun/trace_gen_quad_form.hpp>
#include <stan/math/prim/mat/err/check_multiplicable.hpp>
#include <stan/math/prim/mat/err/check_square.hpp>

namespace stan {
  namespace math {
    namespace {
      template <typename TD, int RD, int CD,
                typename TA, int RA, int CA,
                typename TB, int RB, int CB>
      class trace_gen_quad_form_vari_alloc : public chainable_alloc {
      public:
        trace_gen_quad_form_vari_alloc(const Eigen::Matrix<TD, RD, CD>& D,
                                       const Eigen::Matrix<TA, RA, CA>& A,
                                       const Eigen::Matrix<TB, RB, CB>& B)
          : D_(D), A_(A), B_(B)
        { }

        double compute() {
          using stan::math::value_of;
          return stan::math::trace_gen_quad_form(value_of(D_),
                                                 value_of(A_),
                                                 value_of(B_));
        }

        Eigen::Matrix<TD, RD, CD>  D_;
        Eigen::Matrix<TA, RA, CA>  A_;
        Eigen::Matrix<TB, RB, CB>  B_;
      };

      template <typename TD, int RD, int CD,
                typename TA, int RA, int CA,
                typename TB, int RB, int CB>
      class trace_gen_quad_form_vari : public vari {
      protected:
        static inline void
        computeAdjoints(const double& adj,
                        const Eigen::Matrix<double, RD, CD>& D,
                        const Eigen::Matrix<double, RA, CA>& A,
                        const Eigen::Matrix<double, RB, CB>& B,
                        Eigen::Matrix<var, RD, CD> *varD,
                        Eigen::Matrix<var, RA, CA> *varA,
                        Eigen::Matrix<var, RB, CB> *varB) {
          Eigen::Matrix<double, CA, CB> AtB;
          Eigen::Matrix<double, RA, CB> BD;
          if (varB || varA)
            BD.noalias() = B*D;
          if (varB || varD)
            AtB.noalias() = A.transpose()*B;

          if (varB) {
            Eigen::Matrix<double, RB, CB> adjB(adj*(A*BD + AtB*D.transpose()));
            for (int j = 0; j < B.cols(); j++)
              for (int i = 0; i < B.rows(); i++)
                (*varB)(i, j).vi_->adj_ += adjB(i, j);
          }
          if (varA) {
            Eigen::Matrix<double, RA, CA> adjA(adj*(B*BD.transpose()));
            for (int j = 0; j < A.cols(); j++)
              for (int i = 0; i < A.rows(); i++)
                (*varA)(i, j).vi_->adj_ += adjA(i, j);
          }
          if (varD) {
            Eigen::Matrix<double, RD, CD> adjD(adj*(B.transpose()*AtB));
            for (int j = 0; j < D.cols(); j++)
              for (int i = 0; i < D.rows(); i++)
                (*varD)(i, j).vi_->adj_ += adjD(i, j);
          }
        }


      public:
        explicit
        trace_gen_quad_form_vari(trace_gen_quad_form_vari_alloc
                                 <TD, RD, CD, TA, RA, CA, TB, RB, CB> *impl)
          : vari(impl->compute()), _impl(impl) { }

        virtual void chain() {
          using stan::math::value_of;
          computeAdjoints(adj_,
                          value_of(_impl->D_),
                          value_of(_impl->A_),
                          value_of(_impl->B_),
                          reinterpret_cast<Eigen::Matrix<var, RD, CD> *>
                          (boost::is_same<TD, var>::value?(&_impl->D_):NULL),
                          reinterpret_cast<Eigen::Matrix<var, RA, CA> *>
                          (boost::is_same<TA, var>::value?(&_impl->A_):NULL),
                          reinterpret_cast<Eigen::Matrix<var, RB, CB> *>
                          (boost::is_same<TB, var>::value?(&_impl->B_):NULL));
        }

        trace_gen_quad_form_vari_alloc<TD, RD, CD, TA, RA, CA, TB, RB, CB>
        *_impl;
      };
    }

    template <typename TD, int RD, int CD,
              typename TA, int RA, int CA,
              typename TB, int RB, int CB>
    inline typename
    boost::enable_if_c< boost::is_same<TD, var>::value ||
    boost::is_same<TA, var>::value ||
    boost::is_same<TB, var>::value,
                        var >::type
      trace_gen_quad_form(const Eigen::Matrix<TD, RD, CD>& D,
                          const Eigen::Matrix<TA, RA, CA>& A,
                          const Eigen::Matrix<TB, RB, CB>& B) {
      stan::math::check_square("trace_gen_quad_form", "A", A);
      stan::math::check_square("trace_gen_quad_form", "D", D);
      stan::math::check_multiplicable("trace_gen_quad_form",
                                      "A", A,
                                      "B", B);
      stan::math::check_multiplicable("trace_gen_quad_form",
                                      "B", B,
                                      "D", D);

      trace_gen_quad_form_vari_alloc<TD, RD, CD, TA, RA, CA, TB, RB, CB>
        *baseVari
        = new trace_gen_quad_form_vari_alloc<TD, RD, CD, TA, RA, CA, TB, RB, CB>
        (D, A, B);

      return var(new trace_gen_quad_form_vari
                 <TD, RD, CD, TA, RA, CA, TB, RB, CB>(baseVari));
    }
  }
}

#endif
