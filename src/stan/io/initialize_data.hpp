#ifndef STAN_IO_INITIALIZE_DATA
#define STAN_IO_INITIALIZE_DATA

#include <stan/math/prim/meta.hpp>
#include <stan/math/memory/stack_alloc.hpp>

namespace stan {
  namespace io {
    namespace internal {
      /**
       * Turn a type into an Eigen Map.
       */
      template <typename T, typename = void>
      struct convert_to_map;
      /**
       * No-op for scalar types
       */
      template <typename T>
      struct convert_to_map<T, require_arithmetic_t<T>> {
        using type = T;
      };

      /**
       * Convert an Eigen type to an `Eigen::Map`
       * @tparam An Eigen type to be wrapped in Map.
       */
      template <typename T>
      struct convert_to_map<T, require_eigen_t<T>> {
        using type = Eigen::Map<plain_type_t<T>>;
      };

      /**
       * Convert an Eigen type and the inner type of std::vectors to Maps.
       * @tparam For Scalars this is a no-op, for Eigen types this holds
       *  a Map of the Eigen type's `plain_type`. For std vectors holding
       *  Eigen types this converts the inner Eigen type into a Map.
       */
      template <typename T>
      using convert_to_map_t = typename convert_to_map<std::decay_t<T>>::type;

      template <typename T>
      struct convert_to_map<T, require_std_vector_t<T>> {
        using type = std::vector<convert_to_map_t<value_type_t<T>>>;
      };
    }
    /**
     * Construct an object onto Stan's stack allocator. For scalars this
     * only returns an NaN.
     * @tparam Ret A user specified return type.
     */
    template <typename Ret, require_arithmetic_t<Ret>* = nullptr>
    inline auto initialize_data(stan::math::stack_alloc& /* allocator */) {
      return std::numeric_limits<Ret>::quiet_NaN();
    }
    /**
     * Construct an `Eigen::Map<Vector>` with data allocated on the given allocator.
     * @tparam Ret A user specified return type with a compile time row or column value of 1.
     * @param allocator The allocator to allocate memory from.
     * @param size The dynamic size of the vector.
     */
    template <typename Ret, require_eigen_vector_t<Ret>* = nullptr>
    inline auto initialize_data(stan::math::stack_alloc& allocator, size_t size) {
      return Eigen::Map<Ret>(allocator.alloc_array<double>(size), size);
    }

    /**
     * Construct an `Eigen::Map<Matrix>` with data allocated on the given allocator.
     * @tparam Ret A user specified return type with Dynamic compile time rows and columns.
     * @param allocator The allocator to allocate memory from.
     * @param size The dynamic size of the vector.
     */
    template <typename Ret, require_eigen_matrix_dynamic_t<Ret>* = nullptr>
    inline auto initialize_data(stan::math::stack_alloc& allocator, size_t rows, size_t cols) {
      return Eigen::Map<Ret>(allocator.alloc_array<double>(rows * cols), rows, cols);
    }

    /**
     * Construct an std::vector with data possibly allocated on the given allocator.
     *  For std vectors holding primitive types this will be allocated on the general
     *   allocator. But for std vectors holding containers this will dynamically
     *   allocate the elements memory on the given allocator.
     * @tparam Ret A std vector.
     * @tparam Dims Parameter pack of integral types.
     * @param allocator The allocator to allocate memory from.
     * @param vec_size The dynamic size of the vector.
     * @param dims A parameter packing holding the sizes of the elements.
     */
    template <typename Ret, require_std_vector_t<Ret>* = nullptr, typename... Dims>
    inline auto initialize_data(stan::math::stack_alloc& allocator, size_t vec_size, Dims... dims) {
      internal::convert_to_map_t<Ret> ret_vec;
      ret_vec.reserve(vec_size);
      for (size_t i = 0; i < vec_size; ++i) {
        ret_vec.push_back(initialize_data<value_type_t<Ret>>(allocator, dims...));
      }
      return ret_vec;
    }
  }
}

#endif
