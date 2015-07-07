#ifndef STAN__IO__TRANSFORMS__VECTOR_SIMPLEX_HPP
#define STAN__IO__TRANSFORMS__VECTOR_SIMPLEX_HPP

#include <stan/math/prim/mat/err/check_simplex.hpp>
#include <stan/math/prim/mat/fun/simplex_constrain.hpp>
#include <stan/math/prim/mat/fun/simplex_free.hpp>
#include <stan/io/transforms/vector_transform.hpp>

namespace stan {
  namespace io {

    template <typename T>
    class vector_simplex: public vector_transform<T> {
    public:
      vector_simplex(size_t N): vector_transform<T>(N) {}

      size_t unconstrained_dim() { return N_ - 1; }

      static void unconstrain(const vector_t& input, std::vector<T>& output) {
        stan::math::check_simplex("stan::io::simplex_unconstrain",
                                  "Vector", input);
        vector_t u = stan::math::simplex_free(input);
        for (idx_t n = 0; n < u.size(); ++n)
          output.push_back(u[n]);
      }

      static std::vector<T> unconstrain(const vector_t& input) {
        std::vector<T> output;
        stan::math::check_simplex("stan::io::simplex_unconstrain",
                                  "Vector", input);
        vector_t u = stan::math::simplex_free(input);
        for (idx_t n = 0; n < u.size(); ++n)
          output.push_back(u[n]);
        return output;
      }

      void constrain(const std::vector<T>& input, vector_t& output) {
        output = stan::prob::simplex_constrain(
                   Eigen::Map<const vector_t>(&(input[0]), input.size()));
      }

      void constrain(const std::vector<T>& input, vector_t& output, T& lp) {
        output = stan::prob::simplex_constrain(
                   Eigen::Map<const vector_t>(&(input[0]), input.size()), lp);
      }

      vector_t constrain(const std::vector<T>& input) {
        // Need to map stl vector to Eigen vector
        return stan::prob::simplex_constrain(
                 Eigen::Map<const vector_t>(&(input[0]), input.size()));
      }

      vector_t constrain(const std::vector<T>& input, T& lp) {
        // Need to map stl vector to Eigen vector
        return stan::prob::simplex_constrain(
                 Eigen::Map<const vector_t>(&(input[0]), input.size()), lp);
      }

    }

  }  // io
}  // stan

#endif
