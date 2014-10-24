#ifndef STAN__AGRAD__REV__VAR_STACK_DEF_HPP
#define STAN__AGRAD__REV__VAR_STACK_DEF_HPP

#include <stan/agrad/rev/var_stack.hpp>

namespace stan {

  namespace agrad {


    // FIXME: manage all this in a thread-local singleton container

    std::vector<chainable*> var_stack_;
    std::vector<chainable*> var_nochain_stack_;
    std::vector<chainable_alloc*> var_alloc_stack_;
    memory::stack_alloc memalloc_;

    std::vector<size_t> nested_var_stack_sizes_;
    std::vector<size_t> nested_var_nochain_stack_sizes_;
    std::vector<size_t> nested_var_alloc_stack_starts_;
  }

}

#endif
