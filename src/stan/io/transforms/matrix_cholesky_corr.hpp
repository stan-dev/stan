#ifndef STAN__IO__TRANSFORMS__MATRIX_CHOLESKY_CORR_HPP
#define STAN__IO__TRANSFORMS__MATRIX_CHOLESKY_CORR_HPP

#include <stan/math/prim/mat/fun/cholesky_corr_free.hpp>
#include <stan/math/prim/mat/fun/cholesky_corr_constrain.hpp>

#include <stan/io/transforms/matrix_transform.hpp>

namespace stan {
  namespace io {

    template <typename T>
    class matrix_cholesky_corr: public matrix_transform<T> {
    private:
      T upper_bound_;

    public:
      matrix_cholesky_corr(idx_t N):
        matrix_transform<T>(N, N) {}

      size_t unconstrained_dim() {
        return (N_ * (N_ - 1)) / 2);
      }

      void unconstrain(const matrix_t& input, std::vector<T>& output) {
        // FIXME:  optimize by unrolling cholesky_factor_free
        Eigen::Matrix<T, Eigen::Dynamic, 1> y_free =
          stan::math::cholesky_corr_free(y);
        for (idx_t n = 0; n < y_free.size(); ++n)
          output.push_back(y_free[n]);
      }

      std::vector<T> unconstrain(const matrix_t& input) {
        std::vector<T> output;

        // FIXME:  optimize by unrolling cholesky_factor_free
        Eigen::Matrix<T, Eigen::Dynamic, 1> y_free =
          stan::math::cholesky_corr_free(y);
        for (idx_t n = 0; n < y_free.size(); ++n)
          output.push_back(y_free[n]);
      }

      void constrain(const std::vector<T>& input, matrix_t& output) {
        output = stan::math::cholesky_corr_constrain(
                   Eigen::Map<const vector_t>
                     (&(input[0]), unconstrained_dim()), N_);
      }

      void constrain(const std::vector<T>& input, matrix_t& output, T& lp) {
        output =  stan::math::cholesky_corr_constrain(
                    Eigen::Map<const vector_t>
                      (&(input[0]), unconstrained_dim()),
                    N_, lp);
      }

      matrix_t constrain(const std::vector<T>& input) {
        return stan::math::cholesky_corr_constrain(
                 Eigen::Map<const vector_t>
                   (&(input[0]), unconstrained_dim()), N_);
      }

      matrix_t constrain(const std::vector<T>& input, T& lp) {
        return stan::math::cholesky_corr_constrain(
                 Eigen::Map<const vector_t>
                   (&(input[0]), unconstrained_dim()),
                 N_, lp);
      }

    }

  }  // io
}  // stan

#endif
