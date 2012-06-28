#ifndef __STAN__AGRAD__PARTIALS_VARI_HPP__
#define __STAN__AGRAD__PARTIALS_VARI_HPP__

#include <stan/agrad/agrad.hpp>
#include <stan/agrad/special_functions.hpp>
#include <stan/meta/traits.hpp>

namespace stan {

  namespace agrad {


    /**
     * A variable implementation that stores a single operand and its
     * derivative with respect to the variable.
     */
    class partials1_vari : public vari {
    private:
      vari* operand1_;
    public: 
      const double partial1_;
      /**
       * Construct a variable implementation with the specified value,
       * operand and derivative.
       * @param value Value of variable.
       * @param operand1 Implementation of first operand.
       * @param partial1 Derivative of variable with respect to first operand.
       */
      partials1_vari(double value, vari* operand1, double partial1) 
        : vari(value), operand1_(operand1), partial1_(partial1) 
      { }
      /**
       * Apply chain rule by incrementing this variable's operand's
       * adjoint by the product of this variable's adjoint and
       * the stored partial.
       */
      void chain() {
        operand1_->adj_ += adj_ * partial1_;
      }
    };

    template <typename T1, typename T2>
    class partials2_vari : public vari {
    private:
      const static size_t nvars_ = !stan::is_constant<T1>::value +
        !stan::is_constant<T2>::value;
      vari* operands_[nvars_];
      double partials_[nvars_];

    public:
      partials2_vari(double value,
                     T1 operand1, double partial1,
                     T2 operand2, double partial2)
        : vari(value) {
        size_t i = 0;
        if (!stan::is_constant<T1>::value) {
          operands_[i] = *(vari**)(&operand1);
          partials_[i] = partial1;
          i++;
        }
        if (!stan::is_constant<T2>::value) {
          operands_[i] = *(vari**)(&operand2);
          partials_[i] = partial2;
        }
      }

      void chain() {
        for (size_t i = 0; i < nvars_; i++)
          operands_[i]->adj_ += adj_ * partials_[i];
      }
    };

    template <typename T1, typename T2, typename T3>
    class partials3_vari : public vari {
    private:
      const static size_t nvars_ = !stan::is_constant<T1>::value +
        !stan::is_constant<T2>::value +
        !stan::is_constant<T3>::value;
      vari* operands_[nvars_];
      double partials_[nvars_];

    public:
      partials3_vari(double value,
                     T1 operand1, double partial1,
                     T2 operand2, double partial2,
                     T3 operand3, double partial3)
        : vari(value) {
        size_t i = 0;
        if (!stan::is_constant<T1>::value) {
          operands_[i] = *(vari**)(&operand1);
          partials_[i] = partial1;
          i++;
        }
        if (!stan::is_constant<T2>::value) {
          operands_[i] = *(vari**)(&operand2);
          partials_[i] = partial2;
          i++;
        }
        if (!stan::is_constant<T3>::value) {
          operands_[i] = *(vari**)(&operand3);
          partials_[i] = partial3;
        }
      }

      void chain() {
        for (size_t i = 0; i < nvars_; i++)
          operands_[i]->adj_ += adj_ * partials_[i];
      }
    };

    class partials1s_vari : public vari {
    private:
      const size_t N_;
      vari** operands1_;   double* partials1_;
    public: 
      partials1s_vari(double value,
                      size_t N,
                      vari** operands1, double* partials1)
        : vari(value),
          N_(N),
          operands1_(operands1),
          partials1_(partials1) { }
      void chain() {
        for (size_t n = 0; n < N_; ++n)
          operands1_[n]->adj_ += adj_ * partials1_[n];
      }
    };


    double extract_vari(double v) { return v; }
    agrad::vari* extract_vari(const var v) { return v.vi_; }

    template<typename T>
    agrad::vari* coerce_to_vari(T x) { return 0; }
    agrad::vari* coerce_to_vari(vari* x) { return x; }

    inline agrad::var simple_var(double v, const agrad::var& y1, double dy1) {
      return agrad::var(new agrad::partials1_vari(v, extract_vari(y1), dy1));
    }

    template <typename T1, typename T2>
    inline agrad::var simple_var(double v,
                                 const T1& y1, double dy1,
                                 const T2& y2, double dy2) {
      return agrad::var(new agrad::partials2_vari<typename var_to_vi<T1>::type,
                        typename var_to_vi<T2>::type>(v,
						      extract_vari(y1), dy1,
						      extract_vari(y2), dy2));
    }

    template <typename T1, typename T2, typename T3>
    inline agrad::var simple_var(double v,
                                 const T1& y1, double dy1,
                                 const T2& y2, double dy2,
                                 const T3& y3, double dy3) {
      return agrad::var(new agrad::partials3_vari<typename var_to_vi<T1>::type,
                        typename var_to_vi<T2>::type,
                        typename var_to_vi<T3>::type>(v,
						      extract_vari(y1), dy1,
						      extract_vari(y2), dy2,
						      extract_vari(y3), dy3));
    }

    inline agrad::var simple_var(double v, size_t N, vari** operands,
                                 double* partials) {
      return var(new agrad::partials1s_vari(v, N, operands, partials));
    }


    namespace {
      template<bool as_double=1>
      struct sanitizer {
        static double sanitize(agrad::var v) { return v.val(); }
      };
      template<>
      struct sanitizer<0> {
        static agrad::var sanitize(agrad::var v) { return v; }
      };
    }

    template<typename T1, typename T2, typename T3>
    struct OperandsAndPartials {
      size_t nvaris;

      agrad::vari** all_varis;
      double* all_partials;

      VectorView<double*, is_vector<T1>::value> d_x1;
      VectorView<double*, is_vector<T2>::value> d_x2;
      VectorView<double*, is_vector<T3>::value> d_x3;

      const static bool all_constant = is_constant<typename is_vector<T1>::type>::value
        && is_constant<typename is_vector<T2>::type>::value
        && is_constant<typename is_vector<T3>::type>::value;

      OperandsAndPartials(const T1& x1, const T2& x2, const T3& x3,
                          VectorView<const T1, is_vector<T1>::value> x1_vec,
                          VectorView<const T2, is_vector<T2>::value> x2_vec,
                          VectorView<const T3, is_vector<T3>::value> x3_vec)
        : nvaris((!is_constant<typename is_vector<T1>::type>::value) * length(x1) +
                 (!is_constant<typename is_vector<T2>::type>::value) * length(x2) +
                 (!is_constant<typename is_vector<T3>::type>::value) * length(x3)),
          all_varis((agrad::vari**)agrad::chainable::operator new(sizeof(agrad::vari*[nvaris]))),
          all_partials((double*)agrad::chainable::operator new(sizeof(double[nvaris]))),
          d_x1(all_partials),
          d_x2(all_partials 
	       + (!is_constant<typename is_vector<T1>::type>::value) * length(x1)),
          d_x3(all_partials 
	       + (!is_constant<typename is_vector<T1>::type>::value) * length(x1)
               + (!is_constant<typename is_vector<T2>::type>::value) * length(x2))
      {
        size_t base = 0;
        if (!is_constant<typename is_vector<T1>::type>::value) {
          for (size_t i = 0; i < length(x1); i++)
            all_varis[base + i] = agrad::coerce_to_vari(agrad::extract_vari(x1_vec[i]));
          base += length(x1);
        }
        if (!is_constant<typename is_vector<T2>::type>::value) {
          for (size_t i = 0; i < length(x2); i++)
            all_varis[base + i] = agrad::coerce_to_vari(agrad::extract_vari(x2_vec[i]));
          base += length(x2);
        }
        if (!is_constant<typename is_vector<T3>::type>::value) {
          for (size_t i = 0; i < length(x3); i++)
            all_varis[base + i] = agrad::coerce_to_vari(agrad::extract_vari(x3_vec[i]));
          base += length(x3);
        }

        for (size_t i = 0; i < nvaris; i++)
          all_partials[i] = 0.0;
      }

      typename boost::math::tools::promote_args<typename is_vector<T1>::type,typename is_vector<T2>::type,typename is_vector<T3>::type>::type
      to_var(double logp) {
        if (all_constant)
          return logp;
        else
          return sanitizer<all_constant>::sanitize(agrad::simple_var(logp, nvaris, all_varis, all_partials));
      }
    };

  } 
} 


#endif
