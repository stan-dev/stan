#ifndef STAN_MATH_REV_MAT_FUN_TRACE_INV_QUAD_FORM_LDLT_HPP
#define STAN_MATH_REV_MAT_FUN_TRACE_INV_QUAD_FORM_LDLT_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/rev/mat/fun/LDLT_alloc.hpp>
#include <stan/math/rev/mat/fun/LDLT_factor.hpp>
#include <boost/utility/enable_if.hpp>
#include <stan/math/prim/mat/err/check_multiplicable.hpp>
#include <stan/math/rev/scal/meta/is_var.hpp>

namespace stan {
  namespace math {
    namespace {
      template <typename T2, int R2, int C2, typename T3, int R3, int C3>
      class trace_inv_quad_form_ldlt_impl : public chainable_alloc {
      protected:
        inline void initializeB(const Eigen::Matrix<var, R3, C3> &B,
                                bool haveD) {
          Eigen::Matrix<double, R3, C3> Bd(B.rows(), B.cols());
          _variB.resize(B.rows(), B.cols());
          for (int j = 0; j < B.cols(); j++) {
            for (int i = 0; i < B.rows(); i++) {
              _variB(i, j) = B(i, j).vi_;
              Bd(i, j) = B(i, j).val();
            }
          }
          AinvB_ = _ldlt.solve(Bd);
          if (haveD)
            C_.noalias() = Bd.transpose()*AinvB_;
          else
            _value = (Bd.transpose()*AinvB_).trace();
        }
        inline void initializeB(const Eigen::Matrix<double, R3, C3> &B,
                                bool haveD) {
          AinvB_ = _ldlt.solve(B);
          if (haveD)
            C_.noalias() = B.transpose()*AinvB_;
          else
            _value = (B.transpose()*AinvB_).trace();
        }

        template<int R1, int C1>
        inline void initializeD(const Eigen::Matrix<var, R1, C1> &D) {
          D_.resize(D.rows(), D.cols());
          _variD.resize(D.rows(), D.cols());
          for (int j = 0; j < D.cols(); j++) {
            for (int i = 0; i < D.rows(); i++) {
              _variD(i, j) = D(i, j).vi_;
              D_(i, j) = D(i, j).val();
            }
          }
        }
        template<int R1, int C1>
        inline void initializeD(const Eigen::Matrix<double, R1, C1> &D) {
          D_ = D;
        }

      public:
        template<typename T1, int R1, int C1>
        trace_inv_quad_form_ldlt_impl(const Eigen::Matrix<T1, R1, C1> &D,
                                      const stan::math::LDLT_factor<T2, R2, C2>
                                      &A,
                                      const Eigen::Matrix<T3, R3, C3> &B)
          : Dtype_(stan::is_var<T1>::value),
            _ldlt(A) {
          initializeB(B, true);
          initializeD(D);

          _value = (D_*C_).trace();
        }

        trace_inv_quad_form_ldlt_impl(const stan::math::LDLT_factor<T2, R2, C2>
                                      &A,
                                      const Eigen::Matrix<T3, R3, C3> &B)
          : Dtype_(2),
            _ldlt(A) {
          initializeB(B, false);
        }

        const int Dtype_;  // 0 = double, 1 = var, 2 = missing
        stan::math::LDLT_factor<T2, R2, C2> _ldlt;
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> D_;
        Eigen::Matrix<vari*, Eigen::Dynamic, Eigen::Dynamic> _variD;
        Eigen::Matrix<vari*, R3, C3> _variB;
        Eigen::Matrix<double, R3, C3> AinvB_;
        Eigen::Matrix<double, C3, C3> C_;
        double _value;
      };

      template <typename T2, int R2, int C2, typename T3, int R3, int C3>
      class trace_inv_quad_form_ldlt_vari : public vari {
      protected:
        static inline
        void
        chainA(const double &adj,
               trace_inv_quad_form_ldlt_impl<double, R2, C2, T3, R3, C3>
               *impl) {
        }
        static inline
        void
        chainB(const double &adj,
               trace_inv_quad_form_ldlt_impl<T2, R2, C2, double, R3, C3>
               *impl) {
        }

        static inline
        void
        chainA(const double &adj,
               trace_inv_quad_form_ldlt_impl<var, R2, C2, T3, R3, C3> *impl) {
          Eigen::Matrix<double, R2, C2> aA;

          if (impl->Dtype_ != 2)
            aA.noalias() = -adj * (impl->AinvB_ * impl->D_.transpose()
                                   * impl->AinvB_.transpose());
          else
            aA.noalias() = -adj*(impl->AinvB_ * impl->AinvB_.transpose());

          for (int j = 0; j < aA.cols(); j++)
            for (int i = 0; i < aA.rows(); i++)
              impl->_ldlt._alloc->_variA(i, j)->adj_ += aA(i, j);
        }
        static inline
        void
        chainB(const double &adj,
               trace_inv_quad_form_ldlt_impl<T2, R2, C2, var, R3, C3> *impl) {
          Eigen::Matrix<double, R3, C3> aB;

          if (impl->Dtype_ != 2)
            aB.noalias() = adj*impl->AinvB_*(impl->D_ + impl->D_.transpose());
          else
            aB.noalias() = 2*adj*impl->AinvB_;

          for (int j = 0; j < aB.cols(); j++)
            for (int i = 0; i < aB.rows(); i++)
              impl->_variB(i, j)->adj_ += aB(i, j);
        }

      public:
        explicit trace_inv_quad_form_ldlt_vari
        (trace_inv_quad_form_ldlt_impl<T2, R2, C2, T3, R3, C3> *impl)
          : vari(impl->_value), _impl(impl)
        { }

        virtual void chain() {
          // F = trace(D * B' * inv(A) * B)
          // aA = -aF * inv(A') * B * D' * B' * inv(A')
          // aB = aF*(inv(A) * B * D + inv(A') * B * D')
          // aD = aF*(B' * inv(A) * B)
          chainA(adj_, _impl);

          chainB(adj_, _impl);

          if (_impl->Dtype_ == 1) {
            for (int j = 0; j < _impl->_variD.cols(); j++)
              for (int i = 0; i < _impl->_variD.rows(); i++)
                _impl->_variD(i, j)->adj_ += adj_*_impl->C_(i, j);
          }
        }

        trace_inv_quad_form_ldlt_impl<T2, R2, C2, T3, R3, C3> *_impl;
      };

    }


    /**
     * Compute the trace of an inverse quadratic form.  I.E., this computes
     *       trace(B^T A^-1 B)
     * where the LDLT_factor of A is provided.
     **/
    template <typename T2, int R2, int C2, typename T3, int R3, int C3>
    inline typename
    boost::enable_if_c<stan::is_var<T2>::value ||
                       stan::is_var<T3>::value,
                       var>::type
      trace_inv_quad_form_ldlt(const stan::math::LDLT_factor<T2, R2, C2> &A,
                               const Eigen::Matrix<T3, R3, C3> &B) {
      stan::math::check_multiplicable("trace_inv_quad_form_ldlt",
                                                "A", A,
                                                "B", B);

      trace_inv_quad_form_ldlt_impl<T2, R2, C2, T3, R3, C3> *_impl
        = new trace_inv_quad_form_ldlt_impl<T2, R2, C2, T3, R3, C3>(A, B);

      return var(new trace_inv_quad_form_ldlt_vari<T2, R2, C2, T3, R3, C3>
                 (_impl));
    }

  }
}

#endif
