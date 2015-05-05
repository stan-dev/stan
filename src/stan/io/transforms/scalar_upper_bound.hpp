#ifndef STAN__IO__SCALAR_UPPER_BOUND_HPP
#define STAN__IO__SCALAR_UPPER_BOUND_HPP

#include <stan/io/transforms/base_transform.hpp>

#include <stan/math/prim/scal/fun/ub_constrain.hpp>
#include <stan/math/prim/scal/fun/ub_free.hpp>

namespace stan {
  namespace io {

    template <double UpperBound, typename T>
    class scalar_upper_bound {
      static void constrain(T input, vector<T>& output) {
        output.push_back(stan::prob::ub_constrain(input, UpperBound))
      }
      
      static void unconstrain(T input, vector<T>& output) {
        output.push_back(stan::prob::ub_free(input, UpperBound));
      }
      
      static int constrained_dim() { return 1; }
      static int unconstrained_dim() { return 1; }
      static std::string base_type() { return "scalar"; }
    }

  }  // io
}  // stan

#endif
