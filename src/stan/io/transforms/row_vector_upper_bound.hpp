#ifndef STAN__IO__TRANSFORMS__ROW_VECTOR_UPPER_BOUND_HPP
#define STAN__IO__TRANSFORMS__ROW_VECTOR_UPPER_BOUND_HPP

#include <stan/math/prim/scal/fun/ub_constrain.hpp>
#include <stan/math/prim/scal/fun/ub_free.hpp>

#include <stan/io/transforms/row_vector_transform.hpp>

namespace stan {
  namespace io {

    template <typename T>
    class row_vector_upper_bound: public row_vector_transform<T> {
    private:
      T upper_bound_;

    public:
      row_vector_upper_bound(T upper_bound, size_t N):
        upper_bound_(upper_bound), vector_transform<T>(N) {}

      size_t unconstrained_dim() { return N_; }

      void unconstrain(const row_vector_t& input, std::vector<T>& output) {
        for (idx_t n = 0; n < input.size(); ++n)
          output.push_back(stan::prob::ub_free(input[n], upper_bound_));
      }

      std::vector<T> unconstrain(const row_vector_t& input) {
        std::vector<T> output;
        for (idx_t n = 0; n < input.size(); ++n)
          output.push_back(stan::prob::ub_free(input[n], upper_bound_));
      }

      void constrain(const std::vector<T>& input, row_vector_t& output) {
        for (idx_t n = 0; n < N_; ++n)
          output(n) = stan::prob::ub_constrain(input[n], upper_bound_);
      }

      void constrain(const std::vector<T>& input, row_vector_t& output, T& lp) {
        for (idx_t n = 0; n < N_; ++n)
          output(n) = stan::prob::ub_constrain(input[n], upper_bound_, lp);
      }

      row_vector_t constrain(const std::vector<T>& input) {
        row_vector_t output(N_);
        for (idx_t n = 0; n < N_; ++n)
          output(n) = stan::prob::ub_constrain(input[n], upper_bound_);
        return output;
      }

      row_vector_t constrain(const std::vector<T>& input, T& lp) {
        row_vector_t output(N_);
        for (idx_t n = 0; n < N_; ++n)
          output(n) = stan::prob::ub_constrain(input[n], upper_bound_, lp);
        return output;
      }

    }

  }  // io
}  // stan

#endif
