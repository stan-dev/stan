#ifndef STAN_RUN_CONFIG_HMC_OUTPUT_HPP
#define STAN_RUN_CONFIG_HMC_OUTPUT_HPP

#include <stan/callbacks/writer.hpp>
#include <stan/callbacks/structured_writer.hpp>
#include <stan/run/defaults_hmc_output.hpp>
#include <memory>
#include <vector>
#include <string>
#include <stdexcept>
#include <algorithm>

namespace stan {
namespace run {

/**
 * Configuration class for HMC output files.
 * This class organizes wirter callbacks for both single-chain and multi-chain
 * sampling runs.
 */
template <typename WriterType = callbacks::writer,
              typename MetricWriterType = callbacks::structured_writer>
class config_hmc_output {
public:
  // config_hmc_output_builder class embedded as friend
  class config_hmc_output_builder {
        friend class config_hmc_output;
        std::vector<WriterType*> init_writers_;
        std::vector<WriterType*> sample_writers_;
        std::vector<WriterType*> diagnostic_writers_;
        std::vector<MetricWriterType*> metric_writers_;

  public:
        config_hmc_output_builder(size_t chains) : 
          init_writers_(chains, nullptr),
          sample_writers_(chains, nullptr),
          diagnostic_writers_(chains, nullptr),
          metric_writers_(chains, nullptr) {
        }

        // Single chain setters
        config_hmc_output_builder& init_writer(WriterType* writer) {
          if (init_writers_.size() != 1) {
            throw std::logic_error("Cannot use single writer setters with multi-chain configuration");
          }
          init_writers_[0] = writer;
          return *this;
        }

        config_hmc_output_builder& sample_writer(WriterType* writer) {
          if (sample_writers_.size() != 1) {
            throw std::logic_error("Cannot use single writer setters with multi-chain configuration");
          }
          sample_writers_[0] = writer;
          return *this;
        }

        config_hmc_output_builder& diagnostic_writer(WriterType* writer) {
          if (diagnostic_writers_.size() != 1) {
            throw std::logic_error("Cannot use single writer setters with multi-chain configuration");
          }
          diagnostic_writers_[0] = writer;
          return *this;
        }

        config_hmc_output_builder& metric_writer(MetricWriterType* writer) {
          if (metric_writers_.size() != 1) {
            throw std::logic_error("Cannot use single writer setters with multi-chain configuration");
          }
          metric_writers_[0] = writer;
          return *this;
        }

        // Multi-chain setters
        config_hmc_output_builder& init_writers(const std::vector<WriterType*>& writers) {
          if (writers.size() != init_writers_.size()) {
            throw std::invalid_argument("Wrong number of output writers");
          }
          init_writers_ = writers;
          return *this;
        }

        config_hmc_output_builder& sample_writers(const std::vector<WriterType*>& writers) {
          if (writers.size() != sample_writers_.size()) {
            throw std::invalid_argument("Wrong number of output writers");
          }
          sample_writers_ = writers;
          return *this;
        }

        config_hmc_output_builder& diagnostic_writers(const std::vector<WriterType*>& writers) {
          if (writers.size() != diagnostic_writers_.size()) {
            throw std::invalid_argument("Wrong number of output writers");
          }
          diagnostic_writers_ = writers;
          return *this;
        }

        config_hmc_output_builder& metric_writers(const std::vector<MetricWriterType*>& writers) {
          if (writers.size() != metric_writers_.size()) {
            throw std::invalid_argument("Wrong number of output writers");
          }
          metric_writers_ = writers;
          return *this;
        }

        config_hmc_output build() {
          validate();
          return config_hmc_output(*this);
        }

        void validate() const {
          for (size_t i = 0; i < sample_writers_.size(); ++i) {
            if (!sample_writers_[i]) {
              throw std::invalid_argument("Sample writer must be set for chain " + std::to_string(i));
            }
          }
        }
  };

  static config_hmc_output_builder create(size_t num_chains) {
        return config_hmc_output_builder(num_chains);
  }

  /**
   * Check if this configuration is for multiple chains.
   *
   * @return true if configured for multiple chains.
   */
  bool is_multichain() const {
        return sample_writers_.size() > 1;
  }

  /**
   * Check if the configuration has metric writers.
   *
   * @return true if metric writers are configured
   */
  bool has_metric_writer() const {
        return std::any_of(metric_writers_.begin(), metric_writers_.end(),
                           [](auto* ptr) { return ptr != nullptr; });
  }

  // Getters

  // Single chain getters (throw if multi-chain)
  WriterType* init_writer() const {
        if (is_multichain()) {
          throw std::logic_error("Cannot use single writer getters with multi-chain configuration");
        }
        return init_writers_[0];
  }
 
  WriterType* sample_writer() const {
        if (is_multichain()) {
          throw std::logic_error("Cannot use single writer getters with multi-chain configuration");
        }
        return sample_writers_[0];
  }
 
  WriterType* diagnostic_writer() const {
        if (is_multichain()) {
          throw std::logic_error("Cannot use single writer getters with multi-chain configuration");
        }
        return diagnostic_writers_[0];
  }
 
  MetricWriterType* metric_writer() const {
        if (is_multichain()) {
          throw std::logic_error("Cannot use single writer getters with multi-chain configuration");
        }
        return metric_writers_[0];
  }

  // Multi-chain getters
  const std::vector<WriterType*>& init_writers() const { return init_writers_; }
  const std::vector<WriterType*>& sample_writers() const { return sample_writers_; }
  const std::vector<WriterType*>& diagnostic_writers() const { return diagnostic_writers_; }
  const std::vector<MetricWriterType*>& metric_writers() const { return metric_writers_; }

private:
  explicit config_hmc_output(const config_hmc_output_builder& builder) : 
        init_writers_(builder.init_writers_),
        sample_writers_(builder.sample_writers_),
        diagnostic_writers_(builder.diagnostic_writers_),
        metric_writers_(builder.metric_writers_) {
  }

  std::vector<WriterType*> init_writers_;
  std::vector<WriterType*> sample_writers_;
  std::vector<WriterType*> diagnostic_writers_;
  std::vector<MetricWriterType*> metric_writers_;
};

}  // namespace run
}  // namespace stan
#endif
