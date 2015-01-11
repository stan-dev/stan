#ifndef STAN__COMMON__PRINT_PROGRESS_HPP
#define STAN__COMMON__PRINT_PROGRESS_HPP

#include <cmath>
#include <iomanip>
#include <stan/common/do_print.hpp>

#include <iostream>

namespace stan {
  namespace common {
    
    void print_progress(const int m, 
                        const int start, 
                        const int finish, 
                        const int refresh, 
                        const bool warmup,
                        const std::string prefix,
                        const std::string suffix,
                        std::ostream& o) {
      int it_print_width = std::ceil(std::log10((double) finish));
      if (do_print(m, (start + m + 1 == finish), refresh)) {
        o << prefix;
        o << "Iteration: ";
        o << std::setw(it_print_width) << m + 1 + start
          << " / " << finish;
        o << " [" << std::setw(3) 
          << static_cast<int>( (100.0 * (start + m + 1)) / finish )
          << "%] ";
        o << (warmup ? " (Warmup)" : " (Sampling)");
        o << suffix;
        o << std::flush;
      }
    }

  } // namespace common

} // namespace stan

#endif
