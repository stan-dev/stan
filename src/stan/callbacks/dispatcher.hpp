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
  virtual ~Channel() {}
};

// Adapter for plain writers.
// These writer types (e.g., stream_writer, unique_stream_writer) support only a
// one-argument operator().
class WriterChannel : public Channel {
 public:
  explicit WriterChannel(stan::callbacks::writer* w) : writer_(w) {
    if (!w)
      throw std::runtime_error("Null writer pointer provided to WriterChannel");
  }
  // Single-argument dispatch: forwards to operator().
  template <typename T>
  void dispatch(const T& value) {
    (*writer_)(value);
  }
  // Plain writers do not support key/value writes.
  void begin_record() {}
  void end_record() {}

 private:
  stan::callbacks::writer* writer_;
};

// Adapter for structured writers.
// The structured_writer interface provides a one-argument write(const
// std::string&) and a key/value write(const std::string&, const T&) overload.
class StructuredWriterChannel : public Channel {
 public:
  explicit StructuredWriterChannel(stan::callbacks::structured_writer* sw)
      : writer_(sw) {
    if (!sw)
      throw std::runtime_error(
          "Null structured writer pointer provided to StructuredWriterChannel");
  }
  // Key dispatch
  void dispatch(const std::string& key) { writer_->write(key); }
  // Key/value dispatch
  template <typename T>
  void dispatch(const std::string& key, const T& value) {
    writer_->write(key, std::forward<T>(value));
  }
  void begin_record() { writer_->begin_record(); }
  void end_record() { writer_->end_record(); }

 private:
  stan::callbacks::structured_writer* writer_;
};

// dispatcher class with two overloads for dispatch().
class dispatcher {
 public:
  dispatcher() = default;
  ~dispatcher() = default;

  void register_channel(InfoType type, std::unique_ptr<Channel> channel) {
    channels_[type] = std::move(channel);
  }

  // Overload for non-string types: only forward to plain writer channels.
  template <typename T, typename = std::enable_if_t<
                            !std::is_same<std::decay_t<T>, std::string>::value>>
  void dispatch(InfoType type, T&& info) {
    auto it = channels_.find(type);
    if (it == channels_.end())
      return;  // silently do nothing
    if (auto* wc = dynamic_cast<WriterChannel*>(it->second.get()))
      wc->dispatch(std::forward<T>(info));
    // We do not forward non-string types to structured writer channels.
  }

  // Overload for string types: forward to both plain and structured writer
  // channels.
  void dispatch(InfoType type, const std::string& info) {
    auto it = channels_.find(type);
    if (it == channels_.end())
      return;  // silently do nothing
    if (auto* wc = dynamic_cast<WriterChannel*>(it->second.get()))
      wc->dispatch(info);
    if (auto* sw = dynamic_cast<StructuredWriterChannel*>(it->second.get()))
      sw->dispatch(info);
  }

  // Forward a begin_record call.
  void begin_record(InfoType type) {
    auto it = channels_.find(type);
    if (it == channels_.end())
      return;
    if (auto* sw = dynamic_cast<StructuredWriterChannel*>(it->second.get()))
      sw->begin_record();
  }

  // Forward an end_record call.
  void end_record(InfoType type) {
    auto it = channels_.find(type);
    if (it == channels_.end())
      return;
    if (auto* sw = dynamic_cast<StructuredWriterChannel*>(it->second.get()))
      sw->end_record();
  }

 protected:
  std::unordered_map<InfoType, std::unique_ptr<Channel>, InfoTypeHash>
      channels_;
};

}  // namespace callbacks
}  // namespace stan

#endif  // STAN_CALLBACKS_DISPATCHER_HPP
