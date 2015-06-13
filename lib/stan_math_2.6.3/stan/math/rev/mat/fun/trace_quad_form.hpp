#ifndef STAN_MATH_REV_MAT_FUN_TRACE_QUAD_FORM_HPP
#define STAN_MATH_REV_MAT_FUN_TRACE_QUAD_FORM_HPP

#include <boost/utility/enable_if.hpp>
#include <boost/type_traits.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <stan/math/rev/scal/fun/value_of.hpp>
#include <stan/math/prim/mat/fun/value_of.hpp>
#include <stan/math/rev/mat/fun/typedefs.hpp>
#include <stan/math/prim/mat/fun/trace_quad_form.hpp>
#include <stan/math/prim/mat/err/check_multiplicable.hpp>
#include <stan/math/prim/mat/err/check_square.hpp>

namespace stan {
  namespace math {
    namespace {
      template <typename TA, int RA, int CA, typename TB, int RB, int CB>
      class trace_quad_form_vari_alloc : public chainable_alloc {
      public:
        trace_quad_form_vari_alloc(const Eigen::Matrix<TA, RA, CA>& A,
                                   const Eigen::Matrix<TB, RB, CB>& B)
        : A_(A), B_(B)
        { }

        double compute() {
          using stan::math::value_of;
          return stan::math::trace_quad_form(value_of(A_),
                                             value_of(B_));
        }

        Eigen::Matrix<TA, RA, CA>  A_;
        Eigen::Matrix<TB, RB, CB>  B_;
      };

      template <typename TA, int RA, int CA, typename TB, int RB, int CB>
      class trace_quad_form_vari : public vari {
      protected:
        static inline void chainA(Eigen::Matrix<double, RA, CA>& A,
                                  const Eigen::Matrix<double, RB, CB>& Bd,
                                  const double& adjC) {}
        static inline void chainB(Eigen::Matrix<double, RB, CB>& B,
                                  const Eigen::Matrix<double, RA, CA>& Ad,
                                  const Eigen::Matrix<double, RB, CB>& Bd,
                                  const double& adjC) {}

        static inline void chainA(Eigen::Matrix<var, RA, CA>& A,
                                  const Eigen::Matrix<double, RB, CB>& Bd,
                                  const double& adjC) {
          Eigen::Matrix<double, RA, CA>     adjA(adjC*Bd*Bd.transpose());
          for (int j = 0; j < A.cols(); j++)
            for (int i = 0; i < A.rows(); i++)
              A(i, j).vi_->adj_ += adjA(i, j);
        }
        static inline void chainB(Eigen::Matrix<var, RB, CB>& B,
                                  const Eigen::Matrix<double, RA, CA>& Ad,
                                  const Eigen::Matrix<double, RB, CB>& Bd,
                                  const double& adjC) {
          Eigen::Matrix<double, RA, CA>     adjB(adjC*(Ad + Ad.transpose())*Bd);
          for (int j = 0; j < B.cols(); j++)
            for (int i = 0; i < B.rows(); i++)
              B(i, j).vi_->adj_ += adjB(i, j);
        }

        inline void chainAB(Eigen::Matrix<TA, RA, CA>& A,
                            Eigen::Matrix<TB, RB, CB>& B,
                            const Eigen::Matrix<double, RA, CA>& Ad,
                            const Eigen::Matrix<double, RB, CB>& Bd,
                            const double& adjC) {
          chainA(A, Bd, adjC);
          chainB(B, Ad, Bd, adjC);
        }


      public:
        explicit
        trace_quad_form_vari
        (trace_quad_form_vari_alloc<TA, RA, CA, TB, RB, CB> *impl)
        : vari(impl->compute()), _impl(impl) { }

        virtual void chain() {
          using stan::math::value_of;
          chainAB(_impl->A_, _impl->B_,
                  value_of(_impl->A_), value_of(_impl->B_),
                  adj_);
        }

        trace_quad_form_vari_alloc<TA, RA, CA, TB, RB, CB> *_impl;
      };
    }

    template <typename TA, int RA, int CA, typename TB, int RB, int CB>
    inline typename
    boost::enable_if_c< boost::is_same<TA, var>::value ||
                        boost::is_same<TB, var>::value,
                        var >::type
    trace_quad_form(const Eigen::Matrix<TA, RA, CA>& A,
                    const Eigen::Matrix<TB, RB, CB>& B) {
      stan::math::check_square("trace_quad_form", "A", A);
      stan::math::check_multiplicable("trace_quad_form",
                                                "A", A,
                                                "B", B);

      trace_quad_form_vari_alloc<TA, RA, CA, TB, RB, CB> *baseVari
        = new trace_quad_form_vari_alloc<TA, RA, CA, TB, RB, CB>(A, B);

      return var(new trace_quad_form_vari<TA, RA, CA, TB, RB, CB>(baseVari));
    }
  }
}

#endif
