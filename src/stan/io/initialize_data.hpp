#ifndef STAN_IO_INITIALIZE_DATA
#define STAN_IO_INITIALIZE_DATA

#include <stan/math/prim/meta.hpp>
#include <stan/math/memory/stack_alloc.hpp>

namespace stan {
  namespace io {
    namespace internal {
      template <typename T, typename = void>
      struct convert_to_map;

      template <typename T>
      struct convert_to_map<T, require_arithmetic_t<T>> {
        using type = T;
      };

      template <typename T>
      struct convert_to_map<T, require_eigen_t<T>> {
        using type = Eigen::Map<T>;
      };

      template <typename T>
      using convert_to_map_t = typename convert_to_map<std::decay_t<T>>::type;

      template <typename T>
      struct convert_to_map<T, require_std_vector_t<T>> {
        using type = std::vector<convert_to_map_t<value_type_t<T>>>;
      };
    }
    template <typename Ret, require_arithmetic_t<Ret>* = nullptr>
    auto initialize_data(const std::string& name, stan::math::stack_alloc& /* allocator */) {
      return std::numeric_limits<Ret>::quiet_NaN();
    }
    template <typename Ret, require_eigen_vector_t<Ret>* = nullptr>
    auto initialize_data(const std::string& name, stan::math::stack_alloc& allocator, size_t size) {
      return Eigen::Map<Ret>(allocator.alloc_array<double>(size), size);
    }

    template <typename Ret, require_eigen_matrix_dynamic_t<Ret>* = nullptr>
    auto initialize_data(const std::string& name, stan::math::stack_alloc& allocator, size_t rows, size_t cols) {
      return Eigen::Map<Ret>(allocator.alloc_array<double>(rows * cols), rows, cols);
    }

    template <typename Ret, require_std_vector_t<Ret>* = nullptr, typename... Dims>
    auto initialize_data(const std::string& name, stan::math::stack_alloc& allocator, size_t vec_size, Dims... dims) {
      internal::convert_to_map_t<Ret> ret_vec;
      ret_vec.reserve(vec_size);
      for (size_t i = 0; i < vec_size; ++i) {
        ret_vec.push_back(initialize_data<value_type_t<Ret>>(name, allocator, dims...));
      }
      return ret_vec;
    }
  }
}

#endif
