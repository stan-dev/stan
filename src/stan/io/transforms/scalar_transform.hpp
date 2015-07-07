#ifndef STAN__IO__TRANSFORMS__SCALAR_TRANSFORM_HPP
#define STAN__IO__TRANSFORMS__SCALAR_TRANSFORM_HPP

#include <vector>
#include <string>

namespace stan {
  namespace io {

    template <typename T>
    class scalar_transform {
    public:

      virtual void unconstrain(T input, std::vector<T>& output) = 0;
      virtual T unconstrain(T input) = 0;

      virtual void constrain(T input, T& output) = 0;
      virtual void constrain(T input, T& output, T& lp) = 0;
      virtual T constrain(T input) = 0;
      virtual T constrain(T input, T& lp) = 0;
    };

  }  // io
}  // stan

#endif
