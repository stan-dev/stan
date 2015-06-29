#ifndef STAN__IO__TRANSFORMS__UNCONSTRAIN__VECTOR_TRANSFORM_HPP
#define STAN__IO__TRANSFORMS__UNCONSTRAIN__VECTOR_TRANSFORM_HPP

#include <string>

#include <stan/math/prim/mat/fun/Eigen.hpp>

namespace stan {
  namespace io {

    template <typename T>
    class vector_transform {
    public:
      typedef Eigen::Matrix<T, Eigen::Dynamic, 1> vector_t;
      typedef typename stan::math::index_type<vector_t>::type idx_t;
      
      virtual int constrained_dim() = 0;
      virtual int unconstrained_dim() = 0;
      std::string base_type() { return "vector"; }
      
      virtual void unconstrain(const vector_t& input, vector<T>& output) = 0;
      virtual vector<T> unconstrain(const vector_t& input) = 0;
      
      virtual void constrain(const vector<T>& input, vector_t& output) = 0;
      virtual void constrain(const vector<T>& input, vector_t& output, T& lp) = 0;
      virtual vector_t constrain(const vector<T>& input) = 0;
      virtual vector_t constrain(const vector<T>& input, T& lp) = 0;
      
    };

  }  // io
}  // stan

#endif
