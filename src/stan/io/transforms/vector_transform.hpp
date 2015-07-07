#ifndef STAN__IO__TRANSFORMS__VECTOR_TRANSFORM_HPP
#define STAN__IO__TRANSFORMS__VECTOR_TRANSFORM_HPP

#include <vector>
#include <string>

#include <stan/math/prim/mat/fun/Eigen.hpp>

namespace stan {
  namespace io {

    template <typename T>
    class vector_transform {
    private:
      size_t N_;

    public:
      vector_transform(size_t N): N_(N) {}

      typedef Eigen::Matrix<T, Eigen::Dynamic, 1> vector_t;
      typedef typename stan::math::index_type<vector_t>::type idx_t;

      size_t size() { return N_; }
      virtual size_t unconstrained_dim() = 0;

      virtual void unconstrain(const vector_t& input,
                               std::vector<T>& output) = 0;
      virtual std::vector<T> unconstrain(const vector_t& input) = 0;

      virtual void constrain(const std::vector<T>& input,
                             vector_t& output) = 0;
      virtual void constrain(const std::vector<T>& input,
                             vector_t& output, T& lp) = 0;
      virtual vector_t constrain(const std::vector<T>& input) = 0;
      virtual vector_t constrain(const std::vector<T>& input, T& lp) = 0;

    };

  }  // io
}  // stan

#endif
