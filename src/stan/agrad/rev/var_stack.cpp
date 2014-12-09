#ifndef __STAN__AGRAD__REV__VAR_STACK_CPP__
#define __STAN__AGRAD__REV__VAR_STACK_CPP__

#include <stan/agrad/rev/var_stack.hpp>
#include <stan/agrad/rev/chainable.hpp>

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

	void clear_stack(std::vector<chainable*> & stack, size_t start) {
		for(size_t it = start; it < stack.size(); ++it)
			delete stack[it];

		if( start != 0 )
			stack.erase( stack.begin() + start, stack.end() );
		else
			stack.clear();
	}
  }
}

#endif
