#ifndef STAN__MATH__REV__ARR__META__GRAD_HPP
#define STAN__MATH__REV__ARR__META__GRAD_HPP

#include <stan/math/rev/arr/meta/chainable.hpp>
#include <stan/math/rev/arr/meta/var_stack.hpp>
#include <vector>

namespace stan {
  namespace agrad {

    /**
     * Compute the gradient for all variables starting from the
     * specified root variable implementation.  Does not recover
     * memory.  This chainable variable's adjoint is initialized using
     * the method <code>init_dependent()</code> and then the chain
     * rule is applied working down the stack from this chainable and
     * calling each chainable's <code>chain()</code> method in turn.
     *
     * <p>This function computes a nested gradient only going back as far
     * as the last nesting.
     *
     * <p>This function does not recover any memory from the computation.
     * 
     * @param vi Variable implementation for root of partial
     * derivative propagation.
     */
    static void grad(chainable* vi) {

      // simple reference implementation (intended as doc):
      //   vi->init_dependent(); 
      //   size_t end = var_stack_.size();
      //   size_t begin = empty_nested() ? 0 : end - nested_size();
      //   for (size_t i = end; --i > begin; )  
      //     var_stack_[i]->chain();

      typedef std::vector<chainable*>::reverse_iterator it_t;
      vi->init_dependent(); 
      it_t begin = ChainableStack::var_stack_.rbegin();
      it_t end = empty_nested() ? ChainableStack::var_stack_.rend() : begin + nested_size();
      for (it_t it = begin; it < end; ++it) {
        (*it)->chain();
      }
    }

    
  }
}

#endif
