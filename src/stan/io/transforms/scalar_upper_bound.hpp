#ifndef STAN__IO__TRANSFORMS__SCALAR_UPPER_BOUND_HPP
#define STAN__IO__TRANSFORMS__SCALAR_UPPER_BOUND_HPP

#include <stan/math/prim/scal/fun/ub_constrain.hpp>
#include <stan/math/prim/scal/fun/ub_free.hpp>

#include <stan/io/transforms/scalar_transform.hpp>

namespace stan {
  namespace io {

    template <typename T>
    class scalar_upper_bound: public scalar_transform<T> {
    private:
      T upper_bound_;

    public:
      scalar_upper_bound(T upper_bound): upper_bound_(upper_bound) {}

      void unconstrain(T input, std::vector<T>& output) {
        output.push_back(stan::prob::ub_free(input, upper_bound_));
      }

      T unconstrain(T input) {
        return stan::prob::ub_free(input, upper_bound_);
      }

      void constrain(T input, T& output) {
        output = stan::math::ub_constrain(input, upper_bound_);
      }

      void constrain(T input, T& output, T& lp) {
        output = stan::math::ub_constrain(input, upper_bound_, lp);
      }

      T constrain(T input) {
        return stan::math::ub_constrain(input, upper_bound_);
      }

      T constrain(T input, T& lp) {
        return stan::math::ub_constrain(input, upper_bound_, lp);
      }

    }

  }  // io
}  // stan

#endif
