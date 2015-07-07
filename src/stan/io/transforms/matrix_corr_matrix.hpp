#ifndef STAN__IO__TRANSFORMS__MATRIX_CORR_MATRIX_HPP
#define STAN__IO__TRANSFORMS__MATRIX_CORR_MATRIX_HPP

#include <stan/math/prim/mat/fun/factor_cov_matrix.hpp>
#include <stan/math/prim/mat/fun/corr_matrix_constrain.hpp>

#include <stan/io/transforms/matrix_transform.hpp>

namespace stan {
  namespace io {

    template <typename T>
    class matrix_corr_matrix: public matrix_transform<T> {
    public:
      typedef Eigen::Array<T, Eigen::Dynamic, 1> array_t;
      static constraint_tolerance() { return 1E-8; }

      matrix_corr_matrix(idx_t N):
        matrix_transform<T>(N, N) {}

      size_t unconstrained_dim() { return (N_ * (N_ - 1)) / 2; }

      void unconstrain(const matrix_t& input, std::vector<T>& output) {
        stan::math::check_corr_matrix("matrix_corr_matrix::unconstrain",
                                      "Matrix", input);

        idx_t N_choose_2 = (N_rows_ * (N_rows_ - 1)) / 2;
        array_t partial_corrs(N_choose_2);
        array_t standard_devs(N_rows_);

        if(!stan::math::factor_cov_matrix(input, partial_corrs, standard_devs))
          throw std::runtime_error("factor_cov_matrix failed"))

        for (idx_t n = 0; n < N_rows_; ++n) {
          // sds on log scale unconstrained
          if (fabs(standard_devs[n] - 0.0) >= constraint_tolerance())
            throw std::runtime_error("sds on log scale are unconstrained");
        }

        for (idx_t n = 0; n < k_choose_2; ++n)
          output.push_back(partial_corrs[n]);
      }

      std::vector<T> unconstrain(const matrix_t& input) {
        std::vector<T> output;

        stan::math::check_corr_matrix("matrix_corr_matrix::unconstrain",
                                      "Matrix", input);

        idx_t N_choose_2 = (N_rows_ * (N_rows_ - 1)) / 2;
        array_t partial_corrs(N_choose_2);
        array_t standard_devs(N_rows_);

        if(!stan::math::factor_cov_matrix(input, partial_corrs, standard_devs))
          throw std::runtime_error("factor_cov_matrix failed"))

        for (idx_t n = 0; n < N_rows_; ++n) {
          // sds on log scale unconstrained
          if (fabs(standard_devs[n] - 0.0) >= constraint_tolerance())
            throw std::runtime_error("sds on log scale are unconstrained");
        }

        for (idx_t n = 0; n < k_choose_2; ++n)
          output.push_back(partial_corrs[n]);
      }

      void constrain(const std::vector<T>& input, matrix_t& output) {
        output = stan::math::corr_matrix_constrain(
                   Eigen::Map<const vector_t>
                     (&(input[0]), unconstrained_dim()));
      }

      void constrain(const std::vector<T>& input, matrix_t& output, T& lp) {
        output =  stan::math::corr_matrix_constrain(
                    Eigen::Map<const vector_t>
                      (&(input[0]), unconstrained_dim()),
                    lp);
      }

      matrix_t constrain(const std::vector<T>& input) {
        return stan::math::corr_matrix_constrain(
                 Eigen::Map<const vector_t>
                   (&(input[0]), unconstrained_dim()));
      }

      matrix_t constrain(const std::vector<T>& input, T& lp) {
        return stan::math::corr_matrix_constrain(
                 Eigen::Map<const vector_t>
                   (&(input[0]), unconstrained_dim()),
                 lp);
      }

    }

  }  // io
}  // stan

#endif
