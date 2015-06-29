#ifndef STAN__IO__TRANSFORMS__UNCONSTRAIN__SCALAR_NOOP_HPP
#define STAN__IO__TRANSFORMS__UNCONSTRAIN__SCALAR_NOOP_HPP

#include <vector>
#include <stan/io/transforms/scalar_transform.hpp>

namespace stan {
  namespace io {

    template <typename T>
    class scalar_noop: public scalar_transform<T> {
    public:
      void unconstrain(T input, vector<T>& output) {
        output.push_back(input);
      }
      
      T unconstrain(T input) {
        return input;
      }
      
      void constrain(T input, T& output) {
        output = input;
      }
      
      void constrain(T input, T& output, T& lp) {
        output = input;
      }
      
      T constrain(T input) {
        return input;
      }
      
      T constrain(T input, T& lp) {
        return input;
      }
    }

  }  // io
}  // stan

#endif
