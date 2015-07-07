#ifndef STAN__IO__TRANSFORMS__SCALAR_CORR_HPP
#define STAN__IO__TRANSFORMS__SCALAR_CORR_HPP

#include <stan/math/prim/scal/fun/corr_constrain.hpp>
#include <stan/math/prim/scal/fun/corr_free.hpp>

#include <stan/io/transforms/scalar_transform.hpp>

namespace stan {
  namespace io {

    template <typename T>
    class scalar_corr: public scalar_transform<T> {
    public:
      void unconstrain(T input, std::vector<T>& output) {
        output.push_back(stan::prob::corr_free(input));
      }

      T unconstrain(T input) {
        return stan::prob::corr_free(input);
      }

      void constrain(T input, T& output) {
        output = stan::prob::corr_constrain(input);
      }

      void constrain(T input, T& output, T& lp) {
        output = stan::prob::corr_constrain(input, lp);
      }

      T constrain(T input) {
        return stan::prob::corr_constrain(input);
      }

      T constrain(T input, T& lp) {
        return stan::prob::corr_constrain(input, lp);
      }
    }

  }  // io
}  // stan

#endif
