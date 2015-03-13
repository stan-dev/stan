#ifndef STAN__SERVICES__MCMC__PROGRESS_HPP
#define STAN__SERVICES__MCMC__PROGRESS_HPP

#include <cmath>
#include <sstream>
#include <iomanip>

namespace stan {
  namespace services {
    namespace mcmc {
    
      std::string progress(const int n,
                           const int start,
                           const int finish,
                           const int refresh,
                           const bool warmup) {
        int it_print_width = std::ceil(std::log10((double) finish));
        
        std::stringstream message;
        message << "Iteration: ";
        message << std::setw(it_print_width) << n + 1 + start
                << " / " << finish;
        message << " [" << std::setw(3)
                << static_cast<int>( (100.0 * (start + n + 1)) / finish )
                << "%] ";
        message << (warmup ? " (Warmup)" : " (Sampling)");
        
        return message.str();
      }

    }
  }
}

#endif
