#ifndef STAN_SERVICES_VARIATIONAL_PRINT_PROGRESS_HPP
#define STAN_SERVICES_VARIATIONAL_PRINT_PROGRESS_HPP

#include <stan/interface_callbacks/writer/base_writer.hpp>
#include <stan/math/prim/scal.hpp>
#include <stan/services/io/do_print.hpp>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>

namespace stan {
  namespace services {
    namespace variational {

      /**
       * Helper function for printing progress for variational inference
       *
       * @param m       total number of iterations
       * @param start   starting iteration
       * @param finish  final iteration
       * @param refresh how frequently we want to print an update
       * @param tune    boolean indicates tuning vs. variational inference
       * @param prefix  prefix string
       * @param suffix  suffix string
       * @param writer  writer
       */
      void print_progress(int m,
                          int start,
                          int finish,
                          int refresh,
                          bool tune,
                          const std::string& prefix,
                          const std::string& suffix,
                          interface_callbacks::writer::base_writer& writer) {
        static const char* function =
          "stan::services::variational::print_progress";

        stan::math::check_positive(function,
                                   "Total number of iterations",
                                   m);
        stan::math::check_nonnegative(function,
                                      "Starting iteration",
                                      start);
        stan::math::check_positive(function,
                                   "Final iteration",
                                   finish);
        stan::math::check_positive(function,
                                   "Refresh rate",
                                   refresh);

        int it_print_width = std::ceil(std::log10(static_cast<double>(finish)));
        if (io::do_print(m - 1, (start + m == finish), refresh)) {
          std::stringstream ss;
          ss << prefix;
          ss << "Iteration: ";
          ss << std::setw(it_print_width) << m + start
             << " / " << finish;
          ss << " [" << std::setw(3)
             << (100 * (start + m)) / finish
             << "%] ";
          ss << (tune ? " (Adaptation)" : " (Variational Inference)");
          ss << suffix;
          writer(ss.str());
        }
      }

    }
  }
}
#endif
