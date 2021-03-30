#ifndef STAN_CALLBACKS_FILE_STREAM_WRITER_HPP
#define STAN_CALLBACKS_FILE_STREAM_WRITER_HPP

#include <stan/callbacks/writer.hpp>
#include <ostream>
#include <vector>
#include <string>

namespace stan {
namespace callbacks {

/**
 * <code>file_stream_writer</code> is an implementation
 * of <code>writer</code> that writes to a file.
 */
class file_stream_writer final : public writer {
 public:
  /**
   * Constructs a file stream writer with an output stream
   * and an optional prefix for comments.
   *
   * @param[in, out] A unique pointer to a type inheriting from `std::ostream`
   * @param[in] comment_prefix string to stream before each comment line.
   *  Default is "".
   */
  explicit file_stream_writer(std::unique_ptr<std::ostream>&& output,
                              const std::string& comment_prefix = "")
      : output_(std::move(output)), comment_prefix_(comment_prefix) {}

  file_stream_writer();
  file_stream_writer(file_stream_writer& other) = delete;
  file_stream_writer(file_stream_writer&& other)
      : output_(std::move(other.output_)),
        comment_prefix_(std::move(other.comment_prefix_)) {}
  /**
   * Virtual destructor
   */
  virtual ~file_stream_writer() {}

  /**
   * Writes a set of names on a single line in csv format followed
   * by a newline.
   *
   * Note: the names are not escaped.
   *
   * @param[in] names Names in a std::vector
   */
  void operator()(const std::vector<std::string>& names) {
    write_vector(names);
  }
  /**
   * Get the underlying stream
   */
  auto& get_stream() { return *output_; }

  /**
   * Writes a set of values in csv format followed by a newline.
   *
   * Note: the precision of the output is determined by the settings
   *  of the stream on construction.
   *
   * @param[in] state Values in a std::vector
   */
  void operator()(const std::vector<double>& state) { write_vector(state); }

  /**
   * Writes the comment_prefix to the stream followed by a newline.
   */
  void operator()() {
    std::stringstream streamer;
    streamer << comment_prefix_ << std::endl;
    *output_ << streamer;
  }

  /**
   * Writes the comment_prefix then the message followed by a newline.
   *
   * @param[in] message A string
   */
  void operator()(const std::string& message) {
    std::stringstream streamer;
    streamer << comment_prefix_ << message << std::endl;
    *output_ << streamer;
  }

 private:
  /**
   * Output stream
   */
  std::unique_ptr<std::ostream> output_;

  /**
   * Comment prefix to use when printing comments: strings and blank lines
   */
  std::string comment_prefix_;

  /**
   * Writes a set of values in csv format followed by a newline.
   *
   * Note: the precision of the output is determined by the settings
   *  of the stream on construction.
   *
   * @param[in] v Values in a std::vector
   */
  template <class T>
  void write_vector(const std::vector<T>& v) {
    if (v.empty())
      return;
    using const_iter = typename std::vector<T>::const_iterator;
    const_iter last = v.end();
    --last;
    std::stringstream streamer;
    for (const_iter it = v.begin(); it != last; ++it) {
           streamer << *it << ",";
    }
    streamer << v.back() << std::endl;
    *output_ << streamer;
  }
};

}  // namespace callbacks
}  // namespace stan

#endif
