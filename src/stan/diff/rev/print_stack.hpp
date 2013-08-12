#ifndef __STAN__DIFF__PRINT_STACK_HPP__
#define __STAN__DIFF__PRINT_STACK_HPP__

#include <ostream>
#include <stan/diff/rev/var_stack.hpp>
#include <stan/diff/rev/vari.hpp>

namespace stan {
  namespace diff {
        
    /** 
     * Prints the auto-dif variable stack. This function
     * is used for debugging purposes.
     * 
     * Only works if all members of stack are vari* as it
     * casts to vari*.  
     * 
     * @param o ostream to modify
     */
    inline void print_stack(std::ostream& o) {
      o << "STACK, size=" << var_stack_.size() << std::endl;
      for (size_t i = 0; i < var_stack_.size(); ++i)
        o << i 
          << "  " << var_stack_[i]
          << "  " << (static_cast<vari*>(var_stack_[i]))->val_
          << " : " << (static_cast<vari*>(var_stack_[i]))->adj_
          << std::endl;
    }

  }
}
#endif
