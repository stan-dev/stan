#ifndef STAN__IO__TRANSFORMS__VECTOR_POSITIVE_ORDERED_HPP
#define STAN__IO__TRANSFORMS__VECTOR_POSITIVE_ORDERED_HPP

#include <stan/math/prim/mat/err/check_positive_ordered.hpp>
#include <stan/math/prim/mat/fun/positive_ordered_constrain.hpp>
#include <stan/io/transforms/vector_transform.hpp>

namespace stan {
  namespace io {

    template <typename T>
    class vector_positive_ordered: public vector_transform<T> {
    public:
      vector_positive_ordered(size_t N): vector_transform<T>(N) {}

      size_t unconstrained_dim() { return N_ - 1; }

      static void unconstrain(const vector_t& input, std::vector<T>& output) {
        // reimplements pos_ordered_free in prob to avoid malloc
        if (input.size() == 0) return;
        stan::math::check_positive_ordered
          ("stan::io::positive_ordered_unconstrain", "Vector", y);
        output.push_back(log(input[0]));
        for (idx_t i = 1; i < y.size(); ++i) {
          output.push_back(log(input[i] - input[i - 1]));
        }
      }

      static std::vector<T> unconstrain(const vector_t& input) {
        std::vector<T> output;
        // reimplements pos_ordered_free in prob to avoid malloc
        if (input.size() == 0) return;
        stan::math::check_positive_ordered
        ("stan::io::positive_ordered_unconstrain", "Vector", y);
        output.push_back(log(input[0]));
        for (idx_t i = 1; i < y.size(); ++i) {
          output.push_back(log(input[i] - input[i - 1]));
        }
      }

      void constrain(const std::vector<T>& input, vector_t& output) {
        output = stan::prob::positive_ordered_constrain(
                   Eigen::Map<const vector_t>(&(input[0]), input.size()));
      }

      void constrain(const std::vector<T>& input, vector_t& output, T& lp) {
        output = stan::prob::positive_ordered_constrain(
                   Eigen::Map<const vector_t>(&(input[0]), input.size()), lp);
      }

      vector_t constrain(const std::vector<T>& input) {
        return stan::prob::positive_ordered_constrain(
                 Eigen::Map<const vector_t>(&(input[0]), input.size()));
      }

      vector_t constrain(const std::vector<T>& input, T& lp) {
        return stan::prob::positive_ordered_constrain(
                 Eigen::Map<const vector_t>(&(input[0]), input.size()), lp);
      }

    }

  }  // io
}  // stan

#endif
