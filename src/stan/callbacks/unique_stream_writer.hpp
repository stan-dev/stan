#ifndef STAN_CALLBACKS_UNIQUE_STREAM_WRITER_HPP
#define STAN_CALLBACKS_UNIQUE_STREAM_WRITER_HPP

#include <stan/callbacks/writer.hpp>
#include <stan/math/prim/fun/Eigen.hpp>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

namespace stan {
namespace callbacks {

/**
 * `unique_stream_writer` is an implementation
 * of `writer` that holds a unique pointer to the stream it is
 * writing to.
 * @tparam Stream A type with with a valid `operator<<(std::string)`
 * @tparam Deleter A class with a valid `operator()` method for deleting the
 * output stream
 */
template <typename Stream, typename Deleter = std::default_delete<Stream>>
class unique_stream_writer final : public writer {
 public:
  /**
   * Constructs a unique stream writer with an output stream
   * and an optional prefix for comments.
   *
   * @param[in, out] output A unique pointer to a type inheriting from
   * `std::ostream`
   * @param[in] comment_prefix string to stream before each comment line.
   *  Default is "".
   */
  explicit unique_stream_writer(std::unique_ptr<Stream, Deleter>&& output,
                                const std::string& comment_prefix = "")
      : output_(std::move(output)), comment_prefix_(comment_prefix) {}

  unique_stream_writer();
  unique_stream_writer(unique_stream_writer& other) = delete;
  unique_stream_writer(unique_stream_writer&& other)
      : output_(std::move(other.output_)),
        comment_prefix_(std::move(other.comment_prefix_)) {}
  /**
   * Virtual destructor
   */
  virtual ~unique_stream_writer() {}

  /**
   * Writes a set of names on a single line in csv format followed
   * by a newline.
   *
   * Note: the names are not escaped.
   *
   * @param[in] names Names in a std::vector
   */
  void operator()(const std::vector<std::string>& names) {
    if (output_ == nullptr)
      return;
    write_vector(names);
  }

  /**
   * Get the underlying stream
   */
  inline auto& get_stream() noexcept { return *output_; }

  /**
   * Writes a set of values in csv format followed by a newline.
   *
   * Note: the precision of the output is determined by the settings
   *  of the stream on construction.
   *
   * @param[in] values Values in a std::vector
   */
  void operator()(const std::vector<double>& values) {
    if (output_ == nullptr)
      return;
    write_vector(values);
  }

  /**
   * Writes multiple rows and columns of values in csv format.
   *
   * Note: the precision of the output is determined by the settings
   *  of the stream on construction.
   *
   * @param[in] values A matrix of values. The input is expected to have
   * parameters in the rows and samples in the columns. The matrix is then
   * transposed for the output.
   */
  void operator()(const Eigen::Ref<Eigen::Matrix<double, -1, -1>>& values) {
    if (output_ == nullptr)
      return;
    *output_ << values.transpose().format(CommaInitFmt);
  }

  /**
   * Writes the comment_prefix to the stream followed by a newline.
   */
  void operator()() {
    if (output_ == nullptr)
      return;
    *output_ << comment_prefix_ << std::endl;
  }

  /**
   * Writes the comment_prefix then the message followed by a newline.
   *
   * @param[in] message A string
   */
  void operator()(const std::string& message) {
    if (output_ == nullptr)
      return;
    *output_ << comment_prefix_ << message << std::endl;
  }

 private:
  /**
   * Comma formatter for writing Eigen matrices
   */
  Eigen::IOFormat CommaInitFmt{
      Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "", "", "\n", "", ""};

  /**
   * Output stream
   */
  std::unique_ptr<Stream, Deleter> output_;

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
    if (output_ == nullptr)
      return;
    if (v.empty()) {
      return;
    }
    auto last = v.end();
    --last;
    for (auto it = v.begin(); it != last; ++it) {
      *output_ << *it << ",";
    }
    *output_ << v.back() << std::endl;
  }
};

}  // namespace callbacks
}  // namespace stan

#endif
