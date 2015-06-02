#ifndef STAN_MATH_REV_CORE_AUTODIFFSTACKSTORAGE_HPP
#define STAN_MATH_REV_CORE_AUTODIFFSTACKSTORAGE_HPP

#include <stan/math/memory/stack_alloc.hpp>
#include <vector>

namespace stan {
  namespace math {

    template<typename ChainableT,
             typename ChainableAllocT>
    struct AutodiffStackStorage {
      static std::vector<ChainableT*> var_stack_;
      static std::vector<ChainableT*> var_nochain_stack_;
      static std::vector<ChainableAllocT*> var_alloc_stack_;
      static stack_alloc memalloc_;

      // nested positions
      static std::vector<size_t> nested_var_stack_sizes_;
      static std::vector<size_t> nested_var_nochain_stack_sizes_;
      static std::vector<size_t> nested_var_alloc_stack_starts_;
    };

    template<typename ChainableT, typename ChainableAllocT>
    std::vector<ChainableT*>
    AutodiffStackStorage<ChainableT, ChainableAllocT>::var_stack_;

    template<typename ChainableT, typename ChainableAllocT>
    std::vector<ChainableT*>
    AutodiffStackStorage<ChainableT, ChainableAllocT>::var_nochain_stack_;

    template<typename ChainableT, typename ChainableAllocT>
    std::vector<ChainableAllocT*>
    AutodiffStackStorage<ChainableT, ChainableAllocT>::var_alloc_stack_;

    template<typename ChainableT, typename ChainableAllocT>
    stack_alloc
    AutodiffStackStorage<ChainableT, ChainableAllocT>::memalloc_;

    template<typename ChainableT, typename ChainableAllocT>
    std::vector<size_t>
    AutodiffStackStorage<ChainableT, ChainableAllocT>::nested_var_stack_sizes_;

    template<typename ChainableT, typename ChainableAllocT>
    std::vector<size_t>
    AutodiffStackStorage<ChainableT, ChainableAllocT>
    ::nested_var_nochain_stack_sizes_;

    template<typename ChainableT, typename ChainableAllocT>
    std::vector<size_t>
    AutodiffStackStorage<ChainableT, ChainableAllocT>
    ::nested_var_alloc_stack_starts_;

  }
}
#endif
