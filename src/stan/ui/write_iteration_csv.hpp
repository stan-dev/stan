#ifndef __STAN__UI__WRITE_ITERATION_CSV_HPP__
#define __STAN__UI__WRITE_ITERATION_CSV_HPP__

#include <ostream>
#include <vector>

namespace stan {
  namespace ui {
    
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
