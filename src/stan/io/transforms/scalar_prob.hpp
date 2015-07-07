#ifndef STAN__IO__TRANSFORMS__SCALAR_PROB_HPP
#define STAN__IO__TRANSFORMS__SCALAR_PROB_HPP

#include <stan/math/prim/scal/fun/prob_constrain.hpp>
#include <stan/math/prim/scal/fun/prob_free.hpp>

#include <stan/io/transforms/scalar_transform.hpp>

namespace stan {
  namespace io {

    template <typename T>
    class scalar_prob: public scalar_transform<T> {
    public:
      void unconstrain(T input, std::vector<T>& output) {
        output.push_back(stan::prob::prob_free(input));
      }

      T unconstrain(T input) {
        stan::prob::prob_free(input);
      }

      void constrain(T input, T& output) {
        output = stan::math::prob_constrain(input);
      }

      void constrain(T input, T& output, T& lp) {
        output = stan::math::prob_constrain(input, lp);
      }

      T constrain(T input) {
        return stan::math::prob_constrain(input);
      }

      T constrain(T input, T& lp) {
        return stan::math::prob_constrain(input, lp);
      }
    }

  }  // io
}  // stan

#endif
