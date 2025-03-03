#ifndef STAN_IO_OUTPUT_CONTROLLER_HPP
#define STAN_IO_OUTPUT_CONTROLLER_HPP

#include <stan/callbacks/writer.hpp>
#include <stan/io/json/json_data.hpp>
#include <stan/io/mcmc_writer.hpp>
#include <memory>
#include <vector>

namespace stan {
namespace io {

class output_controller {
 public:
  output_controller() = default;
  
  // Add a writer for samples
  void add_sample_writer(std::unique_ptr<callbacks::writer> writer) {
    sample_writers_.push_back(std::move(writer));
  }
  
  // Add a writer for diagnostics
  void add_diagnostic_writer(std::unique_ptr<callbacks::writer> writer) {
    diagnostic_writers_.push_back(std::move(writer));
  }
  
  // Write sample data to all registered sample writers
  void write_sample(const std::vector<double>& sample) {
    for (auto& writer : sample_writers_) {
      writer->operator()(sample);
    }
  }
  
  // Write diagnostic data to all registered diagnostic writers
  void write_diagnostic(const std::vector<double>& diagnostic) {
    for (auto& writer : diagnostic_writers_) {
      writer->operator()(diagnostic);
    }
  }

 private:
  std::vector<std::unique_ptr<callbacks::writer>> sample_writers_;
  std::vector<std::unique_ptr<callbacks::writer>> diagnostic_writers_;
};

} // namespace io
} // namespace stan

#endif 