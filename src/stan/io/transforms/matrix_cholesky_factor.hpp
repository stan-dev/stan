#ifndef STAN__IO__TRANSFORMS__MATRIX_CHOLESKY_FACTOR_HPP
#define STAN__IO__TRANSFORMS__MATRIX_CHOLESKY_FACTOR_HPP

#include <stan/math/prim/mat/fun/cholesky_factor_free.hpp>
#include <stan/math/prim/mat/fun/cholesky_factor_constrain.hpp>

#include <stan/io/transforms/matrix_transform.hpp>

namespace stan {
  namespace io {

    template <typename T>
    class matrix_cholesky_factor: public matrix_transform<T> {
    private:
      T upper_bound_;

    public:
      matrix_cholesky_factor(idx_t N_rows, idx_t N_cols):
        matrix_transform<T>(N_rows, N_cols) {}

      size_t unconstrained_dim() {
        return (N_cols_ * (N_cols_ + 1)) / 2 + (N_rows_ - N_cols_) * N_cols_);
      }

      void unconstrain(const matrix_t& input, std::vector<T>& output) {
        // FIXME:  optimize by unrolling cholesky_factor_free
        Eigen::Matrix<T, Eigen::Dynamic, 1> y_free =
          stan::math::cholesky_factor_free(y);
        for (idx_t n = 0; n < y_free.size(); ++n)
          output.push_back(y_free[n]);
      }

      std::vector<T> unconstrain(const matrix_t& input) {
        std::vector<T> output;

        // FIXME:  optimize by unrolling cholesky_factor_free
        Eigen::Matrix<T, Eigen::Dynamic, 1> y_free =
          stan::math::cholesky_factor_free(y);
        for (idx_t n = 0; n < y_free.size(); ++n)
          output.push_back(y_free[n]);
      }

      void constrain(const std::vector<T>& input, matrix_t& output) {
        output = stan::math::cholesky_factor_constrain(
                   Eigen::Map<const vector_t>
                     (&(input[0]), unconstrained_dim()), N_rows_, N_cols_);
      }

      void constrain(const std::vector<T>& input, matrix_t& output, T& lp) {
        output =  stan::math::cholesky_factor_constrain(
                    Eigen::Map<const vector_t>
                      (&(input[0]), unconstrained_dim()),
                    N_rows_, N_cols_, lp);
      }

      matrix_t constrain(const std::vector<T>& input) {
        return stan::math::cholesky_factor_constrain(
                 Eigen::Map<const vector_t>
                   (&(input[0]), unconstrained_dim()), N_rows_, N_cols_);
      }

      matrix_t constrain(const std::vector<T>& input, T& lp) {
        return stan::math::cholesky_factor_constrain(
                 Eigen::Map<const vector_t>
                   (&(input[0]), unconstrained_dim()),
                 N_rows_, N_cols_, lp);
      }

    }

  }  // io
}  // stan

#endif
