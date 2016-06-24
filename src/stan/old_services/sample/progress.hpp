#ifndef STAN_OLD_SERVICES_SAMPLE_PROGRESS_HPP
#define STAN_OLD_SERVICES_SAMPLE_PROGRESS_HPP

#include <stan/old_services/io/do_print.hpp>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>

namespace stan {
  namespace services {
    namespace sample {

      std::string progress(const int m,
                           const int start,
                           const int finish,
                           const int refresh,
                           const bool warmup) {
        int it_print_width = std::ceil(std::log10(static_cast<double>(finish)));
        std::stringstream message;
        message << "Iteration: ";
        message << std::setw(it_print_width) << m + 1 + start
                << " / " << finish;
        message << " [" << std::setw(3)
                << static_cast<int>( (100.0 * (start + m + 1)) / finish )
                << "%] ";
        message << (warmup ? " (Warmup)" : " (Sampling)");

        return message.str();
      }

    }
  }
}

#endif
