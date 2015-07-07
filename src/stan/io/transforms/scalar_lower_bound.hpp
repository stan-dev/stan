#ifndef STAN__IO__TRANSFORMS__SCALAR_LOWER_BOUND_HPP
#define STAN__IO__TRANSFORMS__SCALAR_LOWER_BOUND_HPP

#include <stan/math/prim/scal/fun/lb_constrain.hpp>
#include <stan/math/prim/scal/fun/lb_free.hpp>

#include <stan/io/transforms/scalar_transform.hpp>

namespace stan {
  namespace io {

    template <typename T>
    class scalar_lower_bound: public scalar_transform<T> {
    private:
      T lower_bound_;
    public:
      scalar_lower_bound(T lower_bound): lower_bound_(lower_bound) {}

      void unconstrain(T input, std::vector<T>& output) {
        output.push_back(stan::prob::lb_free(input, lower_bound_));
      }

      T unconstrain(T input) {
        return stan::prob::lb_free(input, lower_bound_);
      }

      void constrain(T input, T& output) {
        output = stan::prob::lb_constrain(input, lower_bound_);
      }

      void constrain(T input, T& output, T& lp) {
        output = stan::prob::lb_constrain(input, lower_bound_, lp);
      }

      T constrain(T input) {
        return stan::prob::lb_constrain(input, lower_bound_);
      }

      T constrain(T input, T& lp) {
        stan::prob::lb_constrain(input, lower_bound_, lp);
      }
    }

  }  // io
}  // stan

#endif
