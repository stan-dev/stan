#ifndef STAN_CALLBACKS_STREAM_WRITER_HPP
#define STAN_CALLBACKS_STREAM_WRITER_HPP

#include <stan/callbacks/writer.hpp>
#include <ostream>
#include <vector>
#include <string>

namespace stan {
namespace callbacks {

/**
 * <code>stream_writer</code> is an implementation
 * of <code>writer</code> that writes to a stream.
 */
class stream_writer : public writer {
 public:
  /**
   * Constructs a stream writer with an output stream
   * and an optional prefix for comments.
   *
   * @param[in, out] output stream to write
   * @param[in] comment_prefix string to stream before
   *   each comment line. Default is "".
   */
  template <typename T, require_t<std::is_same<std::unique_ptr<std::fstream>, std::decay_t<T>>>* = nullptr>
  explicit stream_writer(T&& output,
                         const std::string& comment_prefix = "")
      : output_(std::move(output)), comment_prefix_(comment_prefix) {}

  /**
   * Virtual destructor
   */
  virtual ~stream_writer() {}
  stream_writer();
  stream_writer(stream_writer& other) : output_(std::move(other.output_)) {}
  stream_writer(stream_writer&& other) : output_(std::move(other.output_)) {}
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
  void operator()() { *output_.get() << comment_prefix_ << std::endl; }

  /**
   * Writes the comment_prefix then the message followed by a newline.
   *
   * @param[in] message A string
   */
  void operator()(const std::string& message) {
    *output_.get() << comment_prefix_ << message << std::endl;
  }

  /**
   * Output stream
   */
  std::unique_ptr<std::fstream> output_;

 private:

  /**
   * Comment prefix to use when printing comments: strings and blank lines
   */
  std::string comment_prefix_{"# "};

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

    typename std::vector<T>::const_iterator last = v.end();
    --last;

    for (typename std::vector<T>::const_iterator it = v.begin(); it != last;
         ++it)
    *output_.get() << *it << ",";
    *output_.get() << v.back() << std::endl;
  }
};

}  // namespace callbacks
}  // namespace stan
#endif
