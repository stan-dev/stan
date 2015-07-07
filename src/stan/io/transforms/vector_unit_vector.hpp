#ifndef STAN__IO__TRANSFORMS__VECTOR_UNIT_VECTOR_HPP
#define STAN__IO__TRANSFORMS__VECTOR_UNIT_VECTOR_HPP

#include <stan/math/prim/mat/err/check_unit_vector.hpp>
#include <stan/math/prim/mat/fun/unit_vector_constrain.hpp>
#include <stan/math/prim/mat/fun/unit_vector_free.hpp>
#include <stan/io/transforms/vector_transform.hpp>

namespace stan {
  namespace io {

    template <typename T>
    class vector_unit_vector: public vector_transform<T> {
    public:
      vector_unit_vector(size_t N): vector_transform<T>(N) {}

      size_t unconstrained_dim() { return N_ - 1; }

      static void unconstrain(const vector_t& input, std::vector<T>& output) {
        stan::math::check_simplex("stan::io::unit_vector_unconstrain",
                                  "Vector", input);
        vector_t u = stan::math::unit_vector_free(input);
        for (idx_t n = 0; n < u.size(); ++n)
          output.push_back(u[n]);
      }

      static std::vector<T> unconstrain(const vector_t& input) {
        std::vector<T> output;
        stan::math::check_simplex("stan::io::unit_vector_unconstrain",
                                  "Vector", input);
        vector_t u = stan::math::unit_vector_free(input);
        for (idx_t n = 0; n < u.size(); ++n)
          output.push_back(u[n]);
        return output;
      }

      void constrain(const std::vector<T>& input, vector_t& output) {
        output = stan::prob::unit_vector_constrain(
                   Eigen::Map<const vector_t>(&(input[0]), input.size()));
      }

      void constrain(const std::vector<T>& input, vector_t& output, T& lp) {
        output = stan::prob::unit_vector_constrain(
                   Eigen::Map<const vector_t>(&(input[0]), input.size()), lp);
      }

      vector_t constrain(const std::vector<T>& input) {
        return stan::prob::unit_vector_constrain(
                 Eigen::Map<const vector_t>(&(input[0]), input.size()));
      }

      vector_t constrain(const std::vector<T>& input, T& lp) {
        return stan::prob::unit_vector_constrain(
                 Eigen::Map<const vector_t>(&(input[0]), input.size()), lp);
      }

    }

  }  // io
}  // stan

#endif
