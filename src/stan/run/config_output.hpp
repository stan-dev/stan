#ifndef STAN_RUN_CONFIG_OUTPUT_HPP
#define STAN_RUN_CONFIG_OUTPUT_HPP

#include <stan/callbacks/logger.hpp>
#include <stan/callbacks/writer.hpp>
#include <stan/callbacks/structured_writer.hpp>
#include <stan/run/defaults_output.hpp>
#include <memory>
#include <vector>
#include <string>
#include <stdexcept>
#include <algorithm>

namespace stan {
namespace run {

/**
 * Configuration class for output channels in Stan.
 *
 * This class organizes output streams for both single-chain and multi-chain
 * sampling runs. It handles loggers and writers for samples, diagnostics,
 * initialization values, and metric parameters.
 */
template <typename WriterType = callbacks::writer,
          typename MetricWriterType = callbacks::structured_writer>
class config_output {
public:
  // config_output_builder class embedded as friend
  class config_output_builder {
    friend class config_output;
    num_chains_config num_chains_;
    logger_config logger_;
    std::vector<WriterType*> init_writers_;
    std::vector<WriterType*> sample_writers_;
    std::vector<WriterType*> diagnostic_writers_;
    std::vector<MetricWriterType*> metric_writers_;

  public:
    config_output_builder() : 
      num_chains_(),
      logger_(),
      init_writers_(1, nullptr),
      sample_writers_(1, nullptr),
      diagnostic_writers_(1, nullptr),
      metric_writers_(1, nullptr) {}

    // Set number of chains
    config_output_builder& num_chains(size_t chains) {
      num_chains_ = num_chains_config(chains);
      init_writers_.resize(chains, nullptr);
      sample_writers_.resize(chains, nullptr);
      diagnostic_writers_.resize(chains, nullptr);
      metric_writers_.resize(chains, nullptr);
      return *this;
    }

    // Set logger
    config_output_builder& logger(callbacks::logger* log) {
      logger_ = logger_config(log);
      return *this;
    }

    // Single chain setters
    config_output_builder& init_writer(WriterType* writer) {
      if (num_chains_.value() != 1) {
        throw std::logic_error("Cannot use single writer setters with multi-chain configuration");
      }
      init_writers_[0] = writer;
      return *this;
    }

    config_output_builder& sample_writer(WriterType* writer) {
      if (num_chains_.value() != 1) {
        throw std::logic_error("Cannot use single writer setters with multi-chain configuration");
      }
      sample_writers_[0] = writer;
      return *this;
    }

    config_output_builder& diagnostic_writer(WriterType* writer) {
      if (num_chains_.value() != 1) {
        throw std::logic_error("Cannot use single writer setters with multi-chain configuration");
      }
      diagnostic_writers_[0] = writer;
      return *this;
    }

    config_output_builder& metric_writer(MetricWriterType* writer) {
      if (num_chains_.value() != 1) {
        throw std::logic_error("Cannot use single writer setters with multi-chain configuration");
      }
      metric_writers_[0] = writer;
      return *this;
    }

    // Multi-chain setters
    config_output_builder& init_writers(const std::vector<WriterType*>& writers) {
      if (writers.size() != num_chains_.value()) {
        throw std::invalid_argument("Writer vector sizes must match num_chains");
      }
      init_writers_ = writers;
      return *this;
    }

    config_output_builder& sample_writers(const std::vector<WriterType*>& writers) {
      if (writers.size() != num_chains_.value()) {
        throw std::invalid_argument("Writer vector sizes must match num_chains");
      }
      sample_writers_ = writers;
      return *this;
    }

    config_output_builder& diagnostic_writers(const std::vector<WriterType*>& writers) {
      if (writers.size() != num_chains_.value()) {
        throw std::invalid_argument("Writer vector sizes must match num_chains");
      }
      diagnostic_writers_ = writers;
      return *this;
    }

    config_output_builder& metric_writers(const std::vector<MetricWriterType*>& writers) {
      if (writers.size() != num_chains_.value()) {
        throw std::invalid_argument("Writer vector sizes must match num_chains");
      }
      metric_writers_ = writers;
      return *this;
    }

    // Individual chain setters
    config_output_builder& init_writer(size_t chain_idx, WriterType* writer) {
      if (chain_idx >= num_chains_.value()) {
        throw std::out_of_range("Chain index out of range");
      }
      init_writers_[chain_idx] = writer;
      return *this;
    }

    config_output_builder& sample_writer(size_t chain_idx, WriterType* writer) {
      if (chain_idx >= num_chains_.value()) {
        throw std::out_of_range("Chain index out of range");
      }
      sample_writers_[chain_idx] = writer;
      return *this;
    }

    config_output_builder& diagnostic_writer(size_t chain_idx, WriterType* writer) {
      if (chain_idx >= num_chains_.value()) {
        throw std::out_of_range("Chain index out of range");
      }
      diagnostic_writers_[chain_idx] = writer;
      return *this;
    }

    config_output_builder& metric_writer(size_t chain_idx, MetricWriterType* writer) {
      if (chain_idx >= num_chains_.value()) {
        throw std::out_of_range("Chain index out of range");
      }
      metric_writers_[chain_idx] = writer;
      return *this;
    }

    config_output build() {
      validate();
      return config_output(*this);
    }

    void validate() const {
      // Check that logger is set
      if (logger_.value() == nullptr) {
        throw std::invalid_argument("Logger must be set");
      }

      // Check that all required writers are set for all chains
      for (size_t i = 0; i < num_chains_.value(); ++i) {
        if (!init_writers_[i]) {
          throw std::invalid_argument("Init writer must be set for chain " + std::to_string(i));
        }
        if (!sample_writers_[i]) {
          throw std::invalid_argument("Sample writer must be set for chain " + std::to_string(i));
        }
        if (!diagnostic_writers_[i]) {
          throw std::invalid_argument("Diagnostic writer must be set for chain " + std::to_string(i));
        }
        // Note: metric writer is optional, so no validation check needed
      }
    }
  };

  static config_output_builder create() {
    return config_output_builder();
  }

  /**
   * Check if this configuration is for multiple chains.
   *
   * @return true if configured for multiple chains (num_chains > 1)
   */
  bool is_multichain() const {
    return num_chains() > 1;
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
  size_t num_chains() const { return num_chains_.value(); }
  callbacks::logger* logger() const { return logger_.value(); }

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

  // Individual chain getters
  WriterType* init_writer(size_t chain_idx) const {
    validate_chain_idx(chain_idx);
    return init_writers_[chain_idx];
  }
 
  WriterType* sample_writer(size_t chain_idx) const {
    validate_chain_idx(chain_idx);
    return sample_writers_[chain_idx];
  }
 
  WriterType* diagnostic_writer(size_t chain_idx) const {
    validate_chain_idx(chain_idx);
    return diagnostic_writers_[chain_idx];
  }
 
  MetricWriterType* metric_writer(size_t chain_idx) const {
    validate_chain_idx(chain_idx);
    return metric_writers_[chain_idx];
  }

private:
  /**
   * Validates that a chain index is within bounds.
   */
  void validate_chain_idx(size_t chain_idx) const {
    if (chain_idx >= num_chains()) {
      throw std::out_of_range("Chain index out of range");
    }
  }

  explicit config_output(const config_output_builder& builder) : 
    num_chains_(builder.num_chains_),
    logger_(builder.logger_),
    init_writers_(builder.init_writers_),
    sample_writers_(builder.sample_writers_),
    diagnostic_writers_(builder.diagnostic_writers_),
    metric_writers_(builder.metric_writers_) {
  }

  num_chains_config num_chains_;
  logger_config logger_;
  std::vector<WriterType*> init_writers_;
  std::vector<WriterType*> sample_writers_;
  std::vector<WriterType*> diagnostic_writers_;
  std::vector<MetricWriterType*> metric_writers_;
};

}  // namespace run
}  // namespace stan
#endif
