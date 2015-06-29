#ifndef STAN__IO__TRANSFORMS__UNCONSTRAIN__VECTOR_NOOP_HPP
#define STAN__IO__TRANSFORMS__UNCONSTRAIN__VECTOR_NOOP_HPP

#include <vector>

#include <stan/math/prim/scal/fun/lb_constrain.hpp>
#include <stan/math/prim/scal/fun/lb_free.hpp>

#include <stan/io/transforms/vector_transform.hpp>

namespace stan {
  namespace io {

    template <typename T>
    class vector_noop: public vector_transform<T> {
    private:
      size_t N_;
        
    public:
      vector_noop(size_t N): N_(N) {}
      
      static void unconstrain(const vector_t& input, vector<T>& output) {
        for (idx_t n = 0; n < input.size(); ++n)
          output.push_back(input[n]);
      }
      
      static vector<T> unconstrain(const vector_t& input) {
        vector<T> output;
        for (idx_t n = 0; n < input.size(); ++n)
          output.push_back(input[n]);
      }
      
      void constrain(const vector<T>& input, vector_t& output) {
        // Need to map stl vector to Eigen vector
        output = input;
      }
      
      void constrain(const vector<T>& input, vector_t& output, T& lp) {
        // Need to map stl vector to Eigen vector
        output = input;
      }
      
      vector_t constrain(const vector<T>& input) {
        // Need to map stl vector to Eigen vector
        return input;
      }
      
      vector_t constrain(const vector<T>& input, T& lp) {
        // Need to map stl vector to Eigen vector
        return input;
      }
      
      static int constrained_dim() { return N_; }
      static int unconstrained_dim() { return N_; }
    }

  }  // io
}  // stan

#endif
