#ifndef STAN__IO__TRANSFORMS__MATRIX_UPPER_BOUND_HPP
#define STAN__IO__TRANSFORMS__MATRIX_UPPER_BOUND_HPP

#include <stan/math/prim/scal/fun/ub_constrain.hpp>
#include <stan/math/prim/scal/fun/ub_free.hpp>

#include <stan/io/transforms/matrix_transform.hpp>

namespace stan {
  namespace io {

    template <typename T>
    class matrix_upper_bound: public matrix_transform<T> {
    private:
      T upper_bound_;

    public:
      matrix_upper_bound(T upper_bound, idx_t N_rows, idx_t N_cols):
        matrix_transform<T>(N_rows, N_cols), upper_bound_(upper_bound) {}

      size_t unconstrained_dim() { return N_; }

      void unconstrain(const matrix_t& input, std::vector<T>& output) {
        for (idx_t col = 0; col < N_cols_; ++col)
          for (idx_t row = 0; row < N_rows_; ++row)
            output.push_back(stan::prob::ub_free(input(row, col),
                                                 upper_bound_));
      }

      std::vector<T> unconstrain(const matrix_t& input) {
        std::vector<T> output;
        for (idx_t col = 0; col < N_cols_; ++col)
          for (idx_t row = 0; row < N_rows_; ++row)
            output.push_back(stan::prob::ub_free(input(row, col),
                                                 upper_bound_));
      }

      void constrain(const std::vector<T>& input, matrix_t& output) {
        for (idx_t col = 0; col < N_cols_; ++col)
          for (idx_t row = 0; row < N_rows_; ++row)
            output(row, col) =
              stan::prob::ub_constrain(input[row * N_cols_ + col],
                                       upper_bound_);
      }

      void constrain(const std::vector<T>& input, matrix_t& output, T& lp) {
        for (idx_t col = 0; col < N_cols_; ++col)
          for (idx_t row = 0; row < N_rows_; ++row)
            output(row, col) =
              stan::prob::ub_constrain(input[row * N_cols_ + col],
                                       upper_bound_, lp);
      }

      matrix_t constrain(const std::vector<T>& input) {
        matrix_t output(N_rows_, N_cols_);
        for (idx_t col = 0; col < N_cols_; ++col)
          for (idx_t row = 0; row < N_rows_; ++row)
            output(row, col) =
              stan::prob::ub_constrain(input[row * N_cols_ + col],
                                       upper_bound_);
        return output;
      }

      matrix_t constrain(const std::vector<T>& input, T& lp) {
        matrix_t output(N_rows_, N_cols_);
        for (idx_t col = 0; col < N_cols_; ++col)
          for (idx_t row = 0; row < N_rows_; ++row)
            output(row, col) =
              stan::prob::ub_constrain(input[row * N_cols_ + col],
                                       upper_bound_, lp);
        return output;
      }

    }

  }  // io
}  // stan

#endif
