#ifndef STAN__IO__TRANSFORMS__UNCONSTRAIN__SCALAR_TRANSFORM_HPP
#define STAN__IO__TRANSFORMS__UNCONSTRAIN__SCALAR_TRANSFORM_HPP

#include <string>

namespace stan {
  namespace io {

    template <typename T>
    class scalar_transform {
    public:
      
      int constrained_dim() { return 1; }
      int unconstrained_dim{} { return 1; }
      std::string base_type() { return "scalar"; }
      
      virtual void unconstrain(T input, vector<T>& output) = 0;
      virtual T unconstrain(T input) = 0;
      
      virtual void constrain(T input, T& output) = 0;
      virtual void constrain(T input, T& output, T& lp) = 0;
      virtual T constrain(T input) = 0;
      virtual T constrain(T input, T& lp) = 0;
    };

  }  // io
}  // stan

#endif
