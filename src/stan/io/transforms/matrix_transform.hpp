#ifndef STAN__IO__TRANSFORMS__MATRIX_TRANSFORM_HPP
#define STAN__IO__TRANSFORMS__MATRIX_TRANSFORM_HPP

#include <vector>
#include <string>

#include <stan/math/prim/mat/fun/Eigen.hpp>

namespace stan {
  namespace io {

    template <typename T>
    class matrix_transform {
    private:
      idx_t N_rows_;
      idx_t N_cols_;

    public:
      matrix_transform(idx_t N_rows, idx_t N_cols):
        N_rows_(N_rows), N_cols_(N_cols) {}

      typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> matrix_t;
      typedef typename stan::math::index_type<matrix_t>::type idx_t;

      idx_t n_rows() { return N_rows_; }
      idx_t n_cols() { return N_cols_; }
      virtual size_t unconstrained_dim() = 0;

      virtual void unconstrain(const matrix_t& input,
                               std::vector<T>& output) = 0;
      virtual std::vector<T> unconstrain(const matrix_t& input) = 0;

      virtual void constrain(const std::vector<T>& input,
                             matrix_t& output) = 0;
      virtual void constrain(const std::vector<T>& input,
                             matrix_t& output, T& lp) = 0;
      virtual matrix_t constrain(const std::vector<T>& input) = 0;
      virtual matrix_t constrain(const std::vector<T>& input, T& lp) = 0;

    };

  }  // io
}  // stan

#endif
