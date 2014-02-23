#ifndef __STAN__COMMON__PRINT_PROGRESS_HPP__
#define __STAN__COMMON__PRINT_PROGRESS_HPP__

#include <cmath>
#include <iomanip>
#include <stan/common/do_print.hpp>

// FIXME: this calls std::cout directly.
#include <iostream>

namespace stan {
  namespace common {
    
    void print_progress(const int m, 
                        const int start, 
                        const int finish, 
                        const int refresh, 
                        const bool warmup,
                        std::ostream& o) {
      int it_print_width = std::ceil(std::log10(finish));
      if (do_print(m, (start + m + 1 == finish), refresh)) {
        o << "Iteration: ";
        o << std::setw(it_print_width) << m + 1 + start
          << " / " << finish;
        o << " [" << std::setw(3) 
          << static_cast<int>( (100.0 * (start + m + 1)) / finish )
          << "%] ";
        o << (warmup ? " (Warmup)" : " (Sampling)");
        o << std::endl;
      }
    }

    void print_progress(const int m, 
                        const int start, 
                        const int finish, 
                        const int refresh, 
                        const bool warmup) {
      print_progress(m, start, finish, refresh, warmup,
                     std::cout);
    }


  } // namespace common

} // namespace stan

#endif
