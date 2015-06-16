#ifndef STAN_MATH_PRIM_SCAL_META_OPERANDSANDPARTIALS_HPP
#define STAN_MATH_PRIM_SCAL_META_OPERANDSANDPARTIALS_HPP

#include <stan/math/fwd/core.hpp>
#include <stan/math/fwd/scal/meta/is_fvar.hpp>
#include <stan/math/fwd/scal/meta/partials_type.hpp>
#include <stan/math/prim/scal/meta/is_vector.hpp>
#include <stan/math/prim/mat/meta/is_vector.hpp>
#include <stan/math/prim/arr/meta/is_vector.hpp>
#include <stan/math/prim/arr/meta/length.hpp>
#include <stan/math/prim/mat/meta/length.hpp>
#include <stan/math/prim/scal/meta/length.hpp>
#include <stan/math/prim/mat/meta/is_vector_like.hpp>
#include <stan/math/prim/scal/meta/is_vector_like.hpp>
#include <stan/math/prim/scal/meta/VectorView.hpp>
#include <stan/math/prim/scal/meta/is_constant_struct.hpp>
#include <stan/math/prim/scal/meta/contains_fvar.hpp>
#include <stan/math/prim/scal/meta/partials_return_type.hpp>
#include <stan/math/prim/scal/meta/partials_type.hpp>
#include <stan/math/prim/scal/meta/return_type.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/rev/scal/meta/is_var.hpp>
#include <stan/math/rev/scal/meta/partials_type.hpp>

namespace stan {
  namespace math {

    class partials_vari : public vari {
    private:
      const size_t N_;
      vari** operands_;
      double* partials_;
    public:
      partials_vari(double value,
                    size_t N,
                    vari** operands, double* partials)
        : vari(value),
          N_(N),
          operands_(operands),
          partials_(partials) { }
      void chain() {
        for (size_t n = 0; n < N_; ++n)
          operands_[n]->adj_ += adj_ * partials_[n];
      }
    };

    namespace {
      template<typename T1, typename T2, typename T3,
               bool is_vec = is_vector<T2>::value,
               bool is_const = is_constant_struct<T2>::value>
      struct incr_deriv {
        inline T3 incr(T1 d_x, const T2& x_d ) {
          return 0;
        }
      };
      template<typename T1, typename T2, typename T3>
      struct incr_deriv<T1, T2, T3, false, false> {
        inline T3 incr(T1 d_x, const T2& x_d) {
          return d_x[0]*x_d.d_;
        }
      };
      template<typename T1, typename T2, typename T3>
      struct incr_deriv<T1, T2, T3, true, false> {
        inline T3 incr(T1 d_x, const T2& x_d) {
          T3 temp = 0;
          for (size_t n = 0; n < length(x_d); n++)
            temp += d_x[n] * x_d[n].d_;
          return temp;
        }
      };

      template<typename T_return_type, typename T_partials_return,
               typename T1, typename T2, typename T3, typename T4,
               typename T5, typename T6,
               bool is_fvar = stan::contains_fvar<T_return_type>::value,
               bool is_const = stan::is_constant_struct<T_return_type>::value>
      struct partials_to_var {
        inline
        T_return_type to_var(double logp, size_t /* nvaris */,
                             vari** /* all_varis */,
                             T_partials_return* /* all_partials */,
                             const T1& x1, const T2& x2, const T3& x3,
                             const T4& x4, const T5& x5, const T6& x6,
                             VectorView<T_partials_return,
                             is_vector<T1>::value,
                             is_constant_struct<T1>::value> d_x1,
                             VectorView<T_partials_return,
                             is_vector<T2>::value,
                             is_constant_struct<T2>::value> d_x2,
                             VectorView<T_partials_return,
                             is_vector<T3>::value,
                             is_constant_struct<T3>::value> d_x3,
                             VectorView<T_partials_return,
                             is_vector<T4>::value,
                             is_constant_struct<T4>::value> d_x4,
                             VectorView<T_partials_return,
                             is_vector<T5>::value,
                             is_constant_struct<T5>::value> d_x5,
                             VectorView<T_partials_return,
                             is_vector<T6>::value,
                             is_constant_struct<T6>::value> d_x6) {
          return logp;
        }
      };

      template<typename T_return_type, typename T_partials_return,
               typename T1, typename T2, typename T3, typename T4,
               typename T5, typename T6>
      struct partials_to_var<T_return_type, T_partials_return,
                             T1, T2, T3, T4, T5, T6,
                             false, false> {
        inline T_return_type to_var(T_partials_return logp, size_t nvaris,
                                    vari** all_varis,
                                    T_partials_return* all_partials,
                                    const T1& x1, const T2& x2, const T3& x3,
                                    const T4& x4, const T5& x5, const T6& x6,
                                    VectorView<T_partials_return,
                                    is_vector<T1>::value,
                                    is_constant_struct<T1>::value> d_x1,
                                    VectorView<T_partials_return,
                                    is_vector<T2>::value,
                                    is_constant_struct<T2>::value> d_x2,
                                    VectorView<T_partials_return,
                                    is_vector<T3>::value,
                                    is_constant_struct<T3>::value> d_x3,
                                    VectorView<T_partials_return,
                                    is_vector<T4>::value,
                                    is_constant_struct<T4>::value> d_x4,
                                    VectorView<T_partials_return,
                                    is_vector<T5>::value,
                                    is_constant_struct<T5>::value> d_x5,
                                    VectorView<T_partials_return,
                                    is_vector<T6>::value,
                                    is_constant_struct<T6>::value> d_x6) {
          return var(new partials_vari(logp, nvaris, all_varis,
                                              all_partials));
        }
      };

      template<typename T_return_type, typename T_partials_return,
               typename T1, typename T2, typename T3, typename T4,
               typename T5, typename T6>
      struct partials_to_var<T_return_type, T_partials_return,
                             T1, T2, T3, T4, T5, T6,
                             true, false> {
        inline T_return_type to_var(T_partials_return logp, size_t nvaris,
                                    vari** all_varis,
                                    T_partials_return* all_partials,
                                    const T1& x1, const T2& x2, const T3& x3,
                                    const T4& x4, const T5& x5, const T6& x6,
                                    VectorView<T_partials_return,
                                    is_vector<T1>::value,
                                    is_constant_struct<T1>::value> d_x1,
                                    VectorView<T_partials_return,
                                    is_vector<T2>::value,
                                    is_constant_struct<T2>::value> d_x2,
                                    VectorView<T_partials_return,
                                    is_vector<T3>::value,
                                    is_constant_struct<T3>::value> d_x3,
                                    VectorView<T_partials_return,
                                    is_vector<T4>::value,
                                    is_constant_struct<T4>::value> d_x4,
                                    VectorView<T_partials_return,
                                    is_vector<T5>::value,
                                    is_constant_struct<T5>::value> d_x5,
                                    VectorView<T_partials_return,
                                    is_vector<T6>::value,
                                    is_constant_struct<T6>::value> d_x6) {
          T_partials_return temp_deriv = 0;
          temp_deriv += incr_deriv<VectorView<T_partials_return,
                                              is_vector<T1>::value,
                                              is_constant_struct<T1>::value>,
                                   T1, T_partials_return>().incr(d_x1, x1);
          temp_deriv += incr_deriv<VectorView<T_partials_return,
                                              is_vector<T2>::value,
                                              is_constant_struct<T2>::value>,
                                   T2, T_partials_return>().incr(d_x2, x2);
          temp_deriv += incr_deriv<VectorView<T_partials_return,
                                              is_vector<T3>::value,
                                              is_constant_struct<T3>::value>,
                                   T3, T_partials_return>().incr(d_x3, x3);
          temp_deriv += incr_deriv<VectorView<T_partials_return,
                                              is_vector<T4>::value,
                                              is_constant_struct<T4>::value>,
                                   T4, T_partials_return>().incr(d_x4, x4);
          temp_deriv += incr_deriv<VectorView<T_partials_return,
                                              is_vector<T5>::value,
                                              is_constant_struct<T5>::value>,
                                   T5, T_partials_return>().incr(d_x5, x5);
          temp_deriv += incr_deriv<VectorView<T_partials_return,
                                              is_vector<T6>::value,
                                              is_constant_struct<T6>::value>,
                                   T6, T_partials_return>().incr(d_x6, x6);
          return stan::math::fvar<T_partials_return>(logp, temp_deriv);
        }
      };

      template<typename T,
               bool is_vec = is_vector<T>::value,
               bool is_const = is_constant_struct<T>::value,
               bool contain_fvar = contains_fvar<T>::value>
      struct set_varis {
        inline size_t set(vari** /*varis*/, const T& /*x*/) {
          return 0U;
        }
      };
      template<typename T>
      struct set_varis<T, true, false, false> {
        inline size_t set(vari** varis, const T& x) {
          for (size_t n = 0; n < length(x); n++)
            varis[n] = x[n].vi_;
          return length(x);
        }
      };
      template<typename T>
      struct set_varis<T, true, false, true> {
        inline size_t set(vari** varis, const T& x) {
          for (size_t n = 0; n < length(x); n++)
            varis[n] = 0;
          return length(x);
        }
      };
      template<>
      struct set_varis<var, false, false, false> {
        inline size_t set(vari** varis, const var& x) {
          varis[0] = x.vi_;
          return (1);
        }
      };
    }

    /**
     * A variable implementation that stores operands and
     * derivatives with respect to the variable.
     */
    template<typename T1 = double, typename T2 = double, typename T3 = double,
             typename T4 = double, typename T5 = double, typename T6 = double>
    struct OperandsAndPartials {
      typedef
      typename stan::partials_return_type<T1, T2, T3, T4, T5, T6>::type
      T_partials_return;

      typedef
      typename stan::return_type<T1, T2, T3, T4, T5, T6>::type T_return_type;

      static const bool all_constant = is_constant<T_return_type>::value;
      size_t nvaris;
      vari** all_varis;
      T_partials_return* all_partials;

      VectorView<T_partials_return,
                 is_vector<T1>::value,
                 is_constant_struct<T1>::value> d_x1;
      VectorView<T_partials_return,
                 is_vector<T2>::value,
                 is_constant_struct<T2>::value> d_x2;
      VectorView<T_partials_return,
                 is_vector<T3>::value,
                 is_constant_struct<T3>::value> d_x3;
      VectorView<T_partials_return,
                 is_vector<T4>::value,
                 is_constant_struct<T4>::value> d_x4;
      VectorView<T_partials_return,
                 is_vector<T5>::value,
                 is_constant_struct<T5>::value> d_x5;
      VectorView<T_partials_return,
                 is_vector<T6>::value,
                 is_constant_struct<T6>::value> d_x6;

      OperandsAndPartials(const T1& x1 = 0, const T2& x2 = 0, const T3& x3 = 0,
                          const T4& x4 = 0, const T5& x5 = 0, const T6& x6 = 0)
        : nvaris(!is_constant_struct<T1>::value * length(x1) +
                 !is_constant_struct<T2>::value * length(x2) +
                 !is_constant_struct<T3>::value * length(x3) +
                 !is_constant_struct<T4>::value * length(x4) +
                 !is_constant_struct<T5>::value * length(x5) +
                 !is_constant_struct<T6>::value * length(x6)),
          all_varis(static_cast<vari**>
                    (chainable::operator new
                     (sizeof(vari*) * nvaris))),
          all_partials(static_cast<T_partials_return*>
                       (chainable::operator new
                        (sizeof(T_partials_return) * nvaris))),
          d_x1(all_partials),
          d_x2(all_partials
               + (!is_constant_struct<T1>::value) * length(x1)),
          d_x3(all_partials
               + (!is_constant_struct<T1>::value) * length(x1)
               + (!is_constant_struct<T2>::value) * length(x2)),
          d_x4(all_partials
               + (!is_constant_struct<T1>::value) * length(x1)
               + (!is_constant_struct<T2>::value) * length(x2)
               + (!is_constant_struct<T3>::value) * length(x3)),
          d_x5(all_partials
               + (!is_constant_struct<T1>::value) * length(x1)
               + (!is_constant_struct<T2>::value) * length(x2)
               + (!is_constant_struct<T3>::value) * length(x3)
               + (!is_constant_struct<T4>::value) * length(x4)),
          d_x6(all_partials
               + (!is_constant_struct<T1>::value) * length(x1)
               + (!is_constant_struct<T2>::value) * length(x2)
               + (!is_constant_struct<T3>::value) * length(x3)
               + (!is_constant_struct<T4>::value) * length(x4)
               + (!is_constant_struct<T5>::value) * length(x5)) {
        size_t base = 0;
        if (!is_constant_struct<T1>::value)
          base += set_varis<T1>().set(&all_varis[base], x1);
        if (!is_constant_struct<T2>::value)
          base += set_varis<T2>().set(&all_varis[base], x2);
        if (!is_constant_struct<T3>::value)
          base += set_varis<T3>().set(&all_varis[base], x3);
        if (!is_constant_struct<T4>::value)
          base += set_varis<T4>().set(&all_varis[base], x4);
        if (!is_constant_struct<T5>::value)
          base += set_varis<T5>().set(&all_varis[base], x5);
        if (!is_constant_struct<T6>::value)
          set_varis<T6>().set(&all_varis[base], x6);
        std::fill(all_partials, all_partials+nvaris, 0);
      }

      T_return_type
      to_var(T_partials_return logp,
             const T1& x1 = 0, const T2& x2 = 0, const T3& x3 = 0,
             const T4& x4 = 0, const T5& x5 = 0, const T6& x6 = 0) {
        return partials_to_var
          <T_return_type, T_partials_return, T1,
           T2, T3, T4, T5, T6>().to_var(logp, nvaris, all_varis,
                                        all_partials,
                                        x1, x2, x3, x4, x5, x6, d_x1, d_x2,
                                        d_x3, d_x4, d_x5, d_x6);
      }
    };


  }
}


#endif
