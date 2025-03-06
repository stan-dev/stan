#ifndef STAN_CALLBACKS_DISPATCHER_HPP
#define STAN_CALLBACKS_DISPATCHER_HPP

#include <stan/callbacks/writer.hpp>
#include <stan/callbacks/structured_writer.hpp>
#include <memory>
#include <unordered_map>
#include <string>
#include <vector>
#include <stdexcept>
#include <utility>
#include <type_traits>
#include <variant>

namespace stan {
namespace callbacks {

enum class InfoType {
  CONFIG,  // series of string messages
  SAMPLE,  // draw from posterior
  METRIC,  // struct with kv pairs 'metric_type', 'stepsize', 'inv_metric'
  ALGORITHM_STATE,  // sampler state for returned draw
};

struct InfoTypeHash {
  std::size_t operator()(const InfoType& type) const {
    return std::hash<int>()(static_cast<int>(type));
  }
};

// Base type for type erasure.
class Channel {
 public:
  virtual ~Channel() = default;
};

// Adapter for plain writers.
class WriterChannel : public Channel {
 public:
  explicit WriterChannel(stan::callbacks::writer* w) : writer_(w) {
    if (!w) {
      throw std::runtime_error("config error, null writer");
    }
  }

  // Handle all types that writer supports via operator()
  void dispatch() { (*writer_)(); }
  void dispatch(const std::string& value) { (*writer_)(value); }
  void dispatch(const std::vector<double>& value) { (*writer_)(value); }
  void dispatch(const std::vector<std::string>& value) { (*writer_)(value); }

  // Handle any Eigen Matrix type
  template <int R, int C>
  void dispatch(const Eigen::Matrix<double, R, C>& value) {
    (*writer_)(value);
  }

  // No key-value support for plain writers
  template <typename T>
  void dispatch(const std::string&, const T&) {}

 private:
  stan::callbacks::writer* writer_;
};

// Adapter for structured writers.
class StructuredWriterChannel : public Channel {
 public:
  explicit StructuredWriterChannel(stan::callbacks::structured_writer* sw)
      : writer_(sw) {
    if (!sw)
      throw std::runtime_error("config error, null writer");
  }
  // Forward all key-value calls directly to the writer
  void dispatch(const std::string& key) { writer_->write(key); }
  // Perfect forwarding for any key-value pair
  template <typename T>
  void dispatch(const std::string& key, T&& value) {
    writer_->write(key, std::forward<T>(value));
  }
  void begin_record() { writer_->begin_record(); }
  void begin_record(const std::string& key) { writer_->begin_record(key); }
  void end_record() { writer_->end_record(); }

 private:
  stan::callbacks::structured_writer* writer_;
};

// dispatcher class
class dispatcher {
 public:
  dispatcher() = default;
  ~dispatcher() = default;

  void register_channel(InfoType type, std::unique_ptr<Channel> channel) {
    channels_[type] = std::move(channel);
  }

  // Empty call
  void dispatch(InfoType type) {
    if (auto* wc = find_channel<WriterChannel>(type))
      wc->dispatch();
  }

  // String, vector<double>, vector<string>
  template <
      typename T,
      typename = std::enable_if_t<
          std::is_same_v<
              std::decay_t<T>,
              std::
                  string> || std::is_same_v<std::decay_t<T>, std::vector<double>> || std::is_same_v<std::decay_t<T>, std::vector<std::string>>>>
  void dispatch(InfoType type, T&& value) {
    if (auto* wc = find_channel<WriterChannel>(type))
      wc->dispatch(std::forward<T>(value));
  }

  // Eigen matrix types
  template <int R, int C>
  void dispatch(InfoType type, const Eigen::Matrix<double, R, C>& value) {
    if (auto* wc = find_channel<WriterChannel>(type))
      wc->dispatch(value);
  }

  // Key with no value (null)
  void dispatch(InfoType type, const std::string& key) {
    if (auto* sw = find_channel<StructuredWriterChannel>(type))
      sw->dispatch(key);
  }

  // Key-value pairs (forward to structured writers)
  template <typename T>
  void dispatch(InfoType type, const std::string& key, T&& value) {
    if (auto* sw = find_channel<StructuredWriterChannel>(type))
      sw->dispatch(key, std::forward<T>(value));
  }

  // Record operations
  void begin_record(InfoType type) {
    if (auto* sw = find_channel<StructuredWriterChannel>(type))
      sw->begin_record();
  }

  void begin_record(InfoType type, const std::string& key) {
    if (auto* sw = find_channel<StructuredWriterChannel>(type))
      sw->begin_record(key);
  }

  void end_record(InfoType type) {
    if (auto* sw = find_channel<StructuredWriterChannel>(type))
      sw->end_record();
  }

 private:
  // Helper to find and cast a channel of specific type
  template <typename ChannelType>
  ChannelType* find_channel(InfoType type) {
    auto it = channels_.find(type);
    if (it == channels_.end())
      return nullptr;
    return dynamic_cast<ChannelType*>(it->second.get());
  }

  std::unordered_map<InfoType, std::unique_ptr<Channel>, InfoTypeHash>
      channels_;
};

}  // namespace callbacks
}  // namespace stan

#endif  // STAN_CALLBACKS_DISPATCHER_HPP
