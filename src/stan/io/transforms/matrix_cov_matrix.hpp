#ifndef STAN__IO__TRANSFORMS__MATRIX_COV_MATRIX_HPP
#define STAN__IO__TRANSFORMS__MATRIX_COV_MATRIX_HPP

#include <stan/math/prim/mat/fun/factor_cov_matrix.hpp>
#include <stan/math/prim/mat/fun/cov_matrix_constrain.hpp>

#include <stan/io/transforms/matrix_transform.hpp>

namespace stan {
  namespace io {

    template <typename T>
    class matrix_cov_matrix: public matrix_transform<T> {
    public:
      typedef Eigen::Array<T, Eigen::Dynamic, 1> array_t;

      matrix_cov_matrix(idx_t N):
        matrix_transform<T>(N, N) {}

      size_t unconstrained_dim() { return (N_ * (N_ + 1)) / 2; }

      void unconstrain(const matrix_t& input, std::vector<T>& output) {
        idx_t N_choose_2 = (N_rows_ * (N_rows_ - 1)) / 2;
        array_t partial_corrs(N_choose_2);
        array_t standard_devs(N_rows_);

        if (!stan::math::factor_cov_matrix(input, partial_corrs, standard_devs))
          throw std::runtime_error("factor_cov_matrix failed"));

        for (idx_t n = 0; n < N_choose_2; ++n)
          output.push_back(partial_corrs[i]);
        for (idx_t i = 0; i < N_rows_; ++i)
          output.push_back(stanard_devs[i]);
      }

      std::vector<T> unconstrain(const matrix_t& input) {
        std::vector<T> output;

        idx_t N_choose_2 = (N_rows_ * (N_rows_ - 1)) / 2;
        array_t partial_corrs(N_choose_2);
        array_t standard_devs(N_rows_);

        if (!stan::math::factor_cov_matrix(input, partial_corrs, standard_devs))
          throw std::runtime_error("factor_cov_matrix failed"));

        for (idx_t n = 0; n < N_choose_2; ++n)
          output.push_back(partial_corrs[i]);
        for (idx_t i = 0; i < N_rows_; ++i)
          output.push_back(stanard_devs[i]);
      }

      void constrain(const std::vector<T>& input, matrix_t& output) {
        output = stan::math::cov_matrix_constrain(
                   Eigen::Map<const vector_t>
                     (&(input[0]), unconstrained_dim()));
      }

      void constrain(const std::vector<T>& input, matrix_t& output, T& lp) {
        output =  stan::math::cov_matrix_constrain(
                    Eigen::Map<const vector_t>
                      (&(input[0]), unconstrained_dim()),
                    lp);
      }

      matrix_t constrain(const std::vector<T>& input) {
        return stan::math::cov_matrix_constrain(
                 Eigen::Map<const vector_t>
                   (&(input[0]), unconstrained_dim()));
      }

      matrix_t constrain(const std::vector<T>& input, T& lp) {
        return stan::math::cov_matrix_constrain(
                 Eigen::Map<const vector_t>
                   (&(input[0]), unconstrained_dim()),
                 lp);
      }

    }

  }  // io
}  // stan

#endif
