#ifndef STAN__IO__TRANSFORMS__ROW_VECTOR_TRANSFORM_HPP
#define STAN__IO__TRANSFORMS__ROW_VECTOR_TRANSFORM_HPP

#include <vector>
#include <string>

#include <stan/math/prim/mat/fun/Eigen.hpp>

namespace stan {
  namespace io {

    template <typename T>
    class row_vector_transform {
    private:
      size_t N_;

    public:
      row_vector_transform(size_t N): N_(N) {}

      typedef Eigen::Matrix<T, 1, Eigen::Dynamic> row_vector_t;
      typedef typename stan::math::index_type<vector_t>::type idx_t;

      size_t size() { return N_; }
      virtual size_t unconstrained_dim() = 0;

      virtual void unconstrain(const row_vector_t& input,
                               std::vector<T>& output) = 0;
      virtual std::vector<T> unconstrain(const row_vector_t& input) = 0;

      virtual void constrain(const std::vector<T>& input,
                             row_vector_t& output) = 0;
      virtual void constrain(const std::vector<T>& input,
                             row_vector_t& output, T& lp) = 0;
      virtual row_vector_t constrain(const std::vector<T>& input) = 0;
      virtual row_vector_t constrain(const std::vector<T>& input, T& lp) = 0;

    };

  }  // io
}  // stan

#endif
