#ifndef STAN__COMMON__WRITE_ITERATION_CSV_HPP
#define STAN__COMMON__WRITE_ITERATION_CSV_HPP

#include <ostream>
#include <vector>

namespace stan {
  namespace common {
    
    void write_iteration_csv(std::ostream& output_stream,
                             const double lp,
                             const std::vector<double>& model_values) {
      output_stream << lp;
      for (size_t i = 0; i < model_values.size(); ++i) {
        output_stream << "," << model_values.at(i);
      }
      output_stream << std::endl;
    }
  }
}

#endif
