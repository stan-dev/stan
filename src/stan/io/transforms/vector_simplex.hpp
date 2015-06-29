#ifndef STAN__IO__TRANSFORMS__UNCONSTRAIN__VECTOR_SIMPLEX_HPP
#define STAN__IO__TRANSFORMS__UNCONSTRAIN__VECTOR_SIMPLEX_HPP

#include <vector>

#include <stan/math/prim/mat/err/check_simplex.hpp>
#include <stan/math/prim/mat/fun/simplex_constrain.hpp>
#include <stan/math/prim/mat/fun/simplex_free.hpp>
#include <stan/io/transforms/vector_transform.hpp>

namespace stan {
  namespace io {
    
    template <typename T>
    class vector_simplex: public vector_transform<T> {
    private:
      size_t N_;
      
    public:
      vector_simplex(size_t N): N_(N) {}
      
      static void unconstrain(const vector_t& input, vector<T>& output) {
        stan::math::check_simplex("stan::io::simplex_unconstrain",
                                  "Vector", input);
        vector_t u = stan::math::simplex_free(input);
        for (idx_t n = 0; n < u.size(); ++n)
          output.push_back(u[n]);
      }
      
      static vector<T> unconstrain(const vector_t& input) {
        vector<T> output;
        stan::math::check_simplex("stan::io::simplex_unconstrain",
                                  "Vector", input);
        vector_t u = stan::math::simplex_free(input);
        for (idx_t n = 0; n < u.size(); ++n)
          output.push_back(u[n]);
        return output;
      }
      
      void constrain(const vector<T>& input, vector_t& output) {
        // Need to map stl vector to Eigen vector
        output = stan::prob::simplex_constrain(input);
      }
      
      void constrain(const vector<T>& input, vector_t& output, T& lp) {
        // Need to map stl vector to Eigen vector
        output = stan::prob::simplex_constrain(input, lp);
      }
      
      vector_t constrain(const vector<T>& input) {
        // Need to map stl vector to Eigen vector
        return stan::prob::simplex_constrain(input);
      }
      
      vector_t constrain(const vector<T>& input, T& lp) {
        // Need to map stl vector to Eigen vector
        return stan::prob::simplex_constrain(input, lp);
      }
      
      static int constrained_dim() { return N_; }
      static int unconstrained_dim() { return N_ - 1; }
    }

  }  // io
}  // stan

#endif
