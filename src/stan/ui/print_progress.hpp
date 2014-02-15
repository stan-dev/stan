#ifndef __STAN__UI__PRINT_PROGRESS_HPP__
#define __STAN__UI__PRINT_PROGRESS_HPP__

#include <cmath>
#include <iomanip>
#include <stan/ui/do_print.hpp>

// FIXME: this calls std::cout directly.
#include <iostream>

namespace stan {
  namespace ui {
    
    void print_progress(const int m, 
                        const int start, 
                        const int finish, 
                        const int refresh, 
                        const bool warmup) {
      int it_print_width = std::ceil(std::log10(finish));
      if (do_print(m, (start + m + 1 == finish), refresh)) {
        std::cout << "Iteration: ";
        std::cout << std::setw(it_print_width) << m + 1 + start
                  << " / " << finish;
        std::cout << " [" << std::setw(3) 
                  << static_cast<int>( (100.0 * (start + m + 1)) / finish )
                  << "%] ";
        std::cout << (warmup ? " (Warmup)" : " (Sampling)");
        std::cout << std::endl;
      }
    }

  } // namespace ui

} // namespace stan

#endif
