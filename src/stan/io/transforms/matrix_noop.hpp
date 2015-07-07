#ifndef STAN__IO__TRANSFORMS__MATRIX_NOOP_HPP
#define STAN__IO__TRANSFORMS__MATRIX_NOOP_HPP

#include <stan/math/prim/scal/fun/lb_constrain.hpp>
#include <stan/math/prim/scal/fun/lb_free.hpp>

#include <stan/io/transforms/matrix_transform.hpp>

namespace stan {
  namespace io {

    template <typename T>
    class matrix_noop: public matrix_transform<T> {
    public:
      matrix_noop(idx_t N_rows, idx_t N_cols):
        matrix_transform<T>(N_rows, N_cols) {}

      size_t unconstrained_dim() { return N_; }

      static void unconstrain(const matrix_t& input,
                              std::vector<T>& output) {
        for (idx_t n = 0; n < input.size(); ++n)
          output.push_back(input[n]);
      }

      static std::vector<T> unconstrain(const matrix_t& input) {
        std::vector<T> output;
        for (idx_t n = 0; n < input.size(); ++n)
          output.push_back(input[n]);
      }

      void constrain(const std::vector<T>& input, matrix_t& output) {
        output = Eigen::Map<const matrix_t>(&(input[0]), N_rows_, N_cols_);
      }

      void constrain(const std::vector<T>& input, matrix_t& output, T& lp) {
        output = Eigen::Map<const matrix_t>(&(input[0]), N_rows_, N_cols_);
      }

      matrix_t constrain(const std::vector<T>& input) {
        return Eigen::Map<const matrix_t>(&(input[0]), N_rows_, N_cols_);
      }

      matrix_t constrain(const std::vector<T>& input, T& lp) {
        return Eigen::Map<const matrix_t>(&(input[0]), N_rows_, N_cols_);
      }

    }

  }  // io
}  // stan

#endif
