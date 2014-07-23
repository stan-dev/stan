#ifndef STAN__AGRAD__PARTIALS_VARI_HPP
#define STAN__AGRAD__PARTIALS_VARI_HPP

#include <stan/meta/traits.hpp>
#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/vari.hpp>

namespace stan {
  namespace agrad {

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
      template<typename T>
      T partials_to_var(double logp, size_t /* nvaris */,
                        agrad::vari** /* all_varis */,
                        double* /* all_partials */) {
        return logp;
      }
      template<>
      var partials_to_var<var>(double logp, size_t nvaris,
                               agrad::vari** all_varis,
                               double* all_partials) {
        return var(new agrad::partials_vari(logp, nvaris, all_varis, all_partials));
      }

      template<typename T, 
               bool is_vec = is_vector<T>::value, 
               bool is_const = is_constant_struct<T>::value>
      struct set_varis {
        inline size_t set(agrad::vari** /*varis*/, const T& /*x*/) {
          return 0U;
        }
      };
      template<typename T>
      struct set_varis <T,true,false>{
        inline size_t set(agrad::vari** varis, const T& x) {
          for (size_t n = 0; n < length(x); n++)
            varis[n] = x[n].vi_;
          return length(x);
        }
      };
      template<>
      struct set_varis<agrad::var, false, false> {
        inline size_t set(agrad::vari** varis, const agrad::var& x) {
          varis[0] = x.vi_;
          return (1);
        }
      };
    }

    /**
     * A variable implementation that stores operands and
     * derivatives with respect to the variable.
     */
    template<typename T1=double, typename T2=double, typename T3=double, 
             typename T4=double, typename T5=double, typename T6=double, 
             typename T_return_type=typename return_type<T1,T2,T3,T4,T5,T6>::type>
    struct OperandsAndPartials {
      const static bool all_constant = is_constant<T_return_type>::value;
      size_t nvaris;
      agrad::vari** all_varis;
      double* all_partials;

      VectorView<double*, is_vector<T1>::value, is_constant_struct<T1>::value> d_x1;
      VectorView<double*, is_vector<T2>::value, is_constant_struct<T2>::value> d_x2;
      VectorView<double*, is_vector<T3>::value, is_constant_struct<T3>::value> d_x3;
      VectorView<double*, is_vector<T4>::value, is_constant_struct<T4>::value> d_x4;
      VectorView<double*, is_vector<T5>::value, is_constant_struct<T5>::value> d_x5;
      VectorView<double*, is_vector<T6>::value, is_constant_struct<T6>::value> d_x6;
      
      OperandsAndPartials(const T1& x1=0, const T2& x2=0, const T3& x3=0, 
                          const T4& x4=0, const T5& x5=0, const T6& x6=0)
        : nvaris(!is_constant_struct<T1>::value * length(x1) +
                 !is_constant_struct<T2>::value * length(x2) +
                 !is_constant_struct<T3>::value * length(x3) +
                 !is_constant_struct<T4>::value * length(x4) +
                 !is_constant_struct<T5>::value * length(x5) +
                 !is_constant_struct<T6>::value * length(x6)),
          all_varis((agrad::vari**)agrad::chainable::operator new(sizeof(agrad::vari*) * nvaris)), 
          all_partials((double*)agrad::chainable::operator new(sizeof(double) * nvaris)),
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
               + (!is_constant_struct<T5>::value) * length(x5))
      {
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
      to_var(double logp) {
        return partials_to_var<T_return_type>(logp, nvaris, all_varis, all_partials);
      }
    };

  } 
} 


#endif
