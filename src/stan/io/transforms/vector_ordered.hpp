#ifndef STAN__IO__TRANSFORMS__UNCONSTRAIN__VECTOR_ORDERED_HPP
#define STAN__IO__TRANSFORMS__UNCONSTRAIN__VECTOR_ORDERED_HPP

#include <vector>

#include <stan/math/prim/mat/err/check_ordered.hpp>
#include <stan/math/prim/mat/fun/ordered_constrain.hpp>
#include <stan/io/transforms/vector_transform.hpp>

namespace stan {
  namespace io {
    
    template <typename T>
    class vector_ordered: public vector_transform<T> {
    private:
      size_t N_;
        
    public:
      vector_ordered(size_t N): N_(N) {}
      
      static void unconstrain(const vector_t& input, vector<T>& output) {
        // reimplements pos_ordered_free in prob to avoid malloc
        if (input.size() == 0) return;
        stan::math::check_ordered("stan::io::ordered_unconstrain", "Vector", input);
        output.push_back(input[0]);
        for (idx_t i = 1; i < input.size(); ++i)
          output.push_back(log(input[i] - input[i - 1]));
      }
      
      static vector<T> unconstrain(const vector_t& input) {
        vector<T> output;
        // reimplements pos_ordered_free in prob to avoid malloc
        if (input.size() == 0) return;
        stan::math::check_ordered("stan::io::ordered_unconstrain", "Vector", input);
        output.push_back(input[0]);
        for (idx_t i = 1; i < input.size(); ++i)
          output.push_back(log(input[i] - input[i - 1]));
        return output;
      }
      
      void constrain(const vector<T>& input, vector_t& output) {
        // Need to map stl vector to Eigen vector
        output = stan::prob::ordered_constrain(input);
      }
      
      void constrain(const vector<T>& input, vector_t& output, T& lp) {
        // Need to map stl vector to Eigen vector
        output = stan::prob::ordered_constrain(input, lp);
      }
      
      vector_t constrain(const vector<T>& input) {
        // Need to map stl vector to Eigen vector
        return stan::prob::ordered_constrain(input);
      }
      
      vector_t constrain(const vector<T>& input, T& lp) {
        // Need to map stl vector to Eigen vector
        return stan::prob::ordered_constrain(input, lp);
      }
      
      static int constrained_dim() { return N_; }
      static int unconstrained_dim() { return N_ - 1; }
    }

  }  // io
}  // stan

#endif
