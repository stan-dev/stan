#ifndef STAN_SERVICES_SAMPLE_PROGRESS_HPP
#define STAN_SERVICES_SAMPLE_PROGRESS_HPP

#include <stan/services/io/do_print.hpp>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>

namespace stan {
  namespace services {
    namespace sample {

      void progress(const int m,
                    const int start,
                    const int finish,
                    const int refresh,
                    const bool warmup,
                    const std::string prefix,
                    const std::string suffix,
                    std::ostream& o) {
        int it_print_width = std::ceil(std::log10(static_cast<double>(finish)));
        if (io::do_print(m, (start + m + 1 == finish), refresh)) {
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

    }
  }
}

#endif
