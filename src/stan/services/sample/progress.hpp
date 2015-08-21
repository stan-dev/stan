#ifndef STAN_SERVICES_SAMPLE_PROGRESS_HPP
#define STAN_SERVICES_SAMPLE_PROGRESS_HPP

#include <cmath>
#include <sstream>
#include <iomanip>
#include <string>

namespace stan {
  namespace services {
    namespace sample {
      std::string progress(const int n,
                           const int start,
                           const int finish,
                           const int refresh,
                           const bool warmup) {
        int it_print_width
          = std::ceil(std::log10((static_cast<double>(finish)));

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

    }  // sample
  }  // services
}  // stan

#endif
