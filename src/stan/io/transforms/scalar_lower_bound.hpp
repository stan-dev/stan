#ifndef STAN__IO__SCALAR_LOWER_BOUND_HPP
#define STAN__IO__SCALAR_LOWER_BOUND_HPP

#include <stan/io/transforms/base_transform.hpp>

#include <stan/math/prim/scal/fun/lb_constrain.hpp>
#include <stan/math/prim/scal/fun/lb_free.hpp>

namespace stan {
  namespace io {

    template <double LowerBound, typename T>
    class scalar_lower_bound {
      static void constrain(T input, vector<double>& output) {
        output.push_back(stan::prob::lb_constrain(input, LowerBound))
      }
      
      static void unconstrain(T input, vector<double>& output) {
        output.push_back(stan::prob::lb_free(input, LowerBound));
      }
      
      static int constrained_dim() { return 1; }
      static int unconstrained_dim() { return 1; }
      static std::string base_type() { return "scalar"; }
    }

  }  // io
}  // stan

#endif
