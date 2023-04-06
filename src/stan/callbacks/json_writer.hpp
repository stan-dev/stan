#ifndef STAN_CALLBACKS_JSON_WRITER_HPP
#define STAN_CALLBACKS_JSON_WRITER_HPP

#include <stan/callbacks/structured_writer.hpp>
#include <stan/math/prim/fun/Eigen.hpp>
#include <vector>
#include <string>

namespace stan {
namespace callbacks {

/**
 * <code>json_writer</code> is an implementation of
 * <code>structured_writer</code> that writes JSON format data to a stream.
 */
template <typename Stream>
class json_writer final : public structured_writer {
 public:
  /**
   * Constructs a json writer with an output stream.
   *
   * @param[in, out] A unique pointer to a type inheriting from `std::ostream`
   */
  explicit json_writer(std::unique_ptr<Stream>&& output)
      : output_(std::move(output)) {}

  json_writer();
  json_writer(json_writer& other) = delete;
  json_writer(json_writer&& other) : output_(std::move(other.output_)) {}

  /**
   * Virtual destructor.
   */
  virtual ~json_writer() {}

  /**
   * Writes "{", initial token of a JSON object.
   */
  void begin() {
    *output_ << "{" << std::endl;
    is_rest_ = false;
  }

  /**
   * Writes object member key followed by ": {"
   *
   */
  void keyed_begin(const std::string& key) {
    write_key(key);
    begin();
  }

  /**
   * Writes "}", final token of a JSON object.
   */
  void end() { *output_ << "}"; }

  void keyed_string(const std::string& key, const std::string& value) {
    write_sep();
    write_key(key);
    *output_ << "\"" << value << "\"";
  }

  void keyed_null(const std::string& key) {
    write_sep();
    write_key(key);
    *output_ << "\"null\" ";
  }

  void keyed_bool(const std::string& key, bool value) {
    write_sep();
    write_key(key);
    *output_ << "\"" << (value ? "true" : "false") << "\" :" << value;
  }

  void keyed_value(const std::string& key, int value) {
    write_sep();
    write_key(key);
    *output_ << value;
  }

  void keyed_value(const std::string& key, double value) {
    write_sep();
    write_key(key);
    *output_ << value;
  }

  void keyed_value(const std::string& key,
                    const std::tuple<Eigen::VectorXd, Eigen::VectorXd>& state) {
    write_sep();
    write_key(key);
    Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols,
                                 ", ", "", "", "\n", "", "");
    *output_ << std::get<0>(state).transpose().eval();
    *output_ << std::get<1>(state).transpose().eval();
  }

  // complex numbers are objects?
  //  void keyed_scalar(const std::string& key, double value) {
  //  write_sep();
  //  write_key(key);
  //    *output_ << "\"" << key << "\" :" << value;
  //  }

  void keyed_values(const std::string& key, const std::vector<double>& values) {
    write_sep();
    write_key(key);
    write_vector(values);
  }

  void keyed_values(const std::string& key, const Eigen::MatrixXd& states) {
    write_sep();
    write_key(key);
    *output_ << "\"" << key << "\" : \"";
    Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols,
                                 ", ", "", "", "\n", "", "");
    *output_ << states.transpose().format(CommaInitFmt);
  }

  /**
   * Writes newline.
   */
  void newline() { *output_ << std::endl; }

  /**
   * Reset state.
   */
  void reset() {
    get_stream().clear();
    is_rest_ = false;
  }

  /**
   * Get the underlying stream
   */
  inline auto& get_stream() noexcept { return *output_; }

 private:
  /**
   * Output stream
   */
  std::unique_ptr<Stream> output_;

  /**
   * State - first member or rest of object?
   */
  bool is_rest_;

  /**
   * Determines whether or not output requires comma separator
   */
  void write_sep() {
    if (is_rest_)
      *output_ << ", ";
    else
      is_rest_ = true;
  }

  /**
   * Writes key plus colon for key-value pair.
   *
   * @param[in] key member name.
   */
  void write_key(const std::string& key) { *output_ << "\"" << key << "\" : "; }

  /**
   * Writes a set of comma separated values.
   *
   * @param[in] v Values in a std::vector
   */
  template <class T>
  void write_vector(const std::vector<T>& v) {
    if (v.empty()) {
      return;
    }
    *output_ << "[ ";
    auto last = v.end();
    --last;
    for (auto it = v.begin(); it != last; ++it) {
      *output_ << *it << ", ";
    }
    *output_ << v.back() << " ]";
  }
};

}  // namespace callbacks
}  // namespace stan
#endif
