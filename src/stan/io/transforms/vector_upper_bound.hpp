#ifndef STAN__IO__TRANSFORMS__UNCONSTRAIN__VECTOR_UPPER_BOUND_HPP
#define STAN__IO__TRANSFORMS__UNCONSTRAIN__VECTOR_UPPER_BOUND_HPP

#include <vector>

#include <stan/math/prim/scal/fun/ub_constrain.hpp>
#include <stan/math/prim/scal/fun/ub_free.hpp>

#include <stan/io/transforms/vector_transform.hpp>

namespace stan {
  namespace io {

    template <typename T>
    class vector_upper_bound: public vector_transform<T> {
    private:
      T upper_bound_;
      size_t N_;
      
    public:
      vector_upper_bound(T upper_bound, size_t N):
        upper_bound_(upper_bound), N_(N) {}
      
      void unconstrain(const vector_t& input, vector<T>& output) {
        for (idx_t n = 0; n < input.size(); ++n)
          output.push_back(stan::prob::ub_free(input[n], upper_bound_));
      }
      
      vector<T> unconstrain(const vector_t& input) {
        vector<T> output;
        for (idx_t n = 0; n < input.size(); ++n)
          output.push_back(stan::prob::ub_free(input[n], upper_bound_));
      }
      
      void constrain(const vector<T>& input, vector_t& output) {
        for (idx_t n = 0; n < N_; ++n)
          o(n) = stan::prob::ub_constrain(input[n], upper_bound_);
      }
      
      void constrain(const vector<T>& input, vector_t& output, T& lp) {
        for (idx_t n = 0; n < N_; ++n)
          o(n) = stan::prob::ub_constrain(input[n], upper_bound_, lp);
      }
      
      vector_t constrain(const vector<T>& input) {
        vector_t output(N_);
        for (idx_t n = 0; n < N_; ++n)
          output(n) = stan::prob::ub_constrain(input[n], upper_bound_);
        return output;
      }
      
      vector_t constrain(const vector<T>& input, T& lp) {
        vector_t output(N_);
        for (idx_t n = 0; n < N_; ++n)
          output(n) = stan::prob::ub_constrain(input[n], upper_bound_, lp);
        return output;
      }
      
      int constrained_dim() { return N_; }
      int unconstrained_dim() { return N_; }
    }

  }  // io
}  // stan

#endif
