#ifndef STAN_CALLBACKS_JSON_WRITER_HPP
#define STAN_CALLBACKS_JSON_WRITER_HPP

#include <stan/callbacks/structured_writer.hpp>
#include <stan/math/prim/fun/Eigen.hpp>
#include <ostream>
#include <string>
#include <vector>

namespace stan {
namespace callbacks {

/**
 * <code>json_writer</code> is an implementation of
 * <code>structured_writer</code> that writes JSON format data to a stream.
 * @tparam Stream A type with with a valid `operator<<(std::string)`
 */
template <typename Stream, typename Deleter = std::default_delete<Stream>>
class json_writer final : public structured_writer {
   private:
  /**
   * Output stream
   */
  std::unique_ptr<Stream, Deleter> output_{nullptr};

  /**
   * State - first member or rest of object?
   */
  bool is_rest_;

 public:
 json_writer() : output_(nullptr) {}
  /**
   * Constructs a json writer with an output stream.
   *
   * @param[in, out] A unique pointer to a type inheriting from `std::ostream`
   */
  explicit json_writer(std::unique_ptr<Stream, Deleter>&& output)
      : output_(std::move(output)) {
    if (output_ == nullptr)
      throw std::invalid_argument("writer cannot be null");
  }

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
   * Writes "}", final token of a JSON object.
   */
  void end() { 
    *output_ << "}";
    write_sep();
    *output_ << "\n";
  }

  void begin_list() {
    *output_ << "[";
    is_rest_ = false;
  }
  void end_list() {
    *output_ << "]";
    write_sep();
    *output_ << "\n";
  }


  /**
   * Writes object member key followed by ": {"
   *
   */
  void write_begin(const std::string& key) {
    write_key(key);
    this->begin();
  }  

  void write(const std::string& key) {
    write_sep();
    write_key(key);
    *output_ << "\"null\" ";
  }


  void write(const std::string& key, const std::string& value) {
    write_sep();
    write_key(key);
    *output_ << "\"" << value << "\"";
  }

  void write(const std::string& key, bool value) {
    write_sep();
    write_key(key);
    *output_ << "\"" << (value ? "true" : "false");
  }

  void write(const std::string& key, int value) {
    write_sep();
    write_key(key);
    *output_ << value;
  }
  // TODO: WRITE THE VIRTUAL FUNCTION FOR THIS
  void write(const std::string& key, std::size_t value) {
    write_sep();
    write_key(key);
    *output_ << value;
  }

  void write(const std::string& key, double value) {
    write_sep();
    write_key(key);
    *output_ << value;
  }

  
  void write(const std::string& key, const std::complex<double>& value) {
    write_sep();
    write_key(key);
      *output_ << "\"" << key << "\" : [" << value.real() << ", " << value.imag() << "]";
    }

  void write(const std::string& key, const std::vector<double>& values) {
    write_sep();
    write_key(key);
    write_vector(values);
  }

  void write(const std::string& key, const std::vector<std::string>& values) {
    write_sep();
    write_key(key);
    write_vector(values);
  }

  void write(const std::string& key, const Eigen::MatrixXd& mat) {
    write_sep();
    write_key(key);
    Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "", "[", "], ", "[", "]");
    *output_ << mat.transpose().format(CommaInitFmt);
  }

  void write(const std::string& key, const Eigen::VectorXd& vec) {
    write_sep();
    write_key(key);
    Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "", "", "", "[", "]");
    *output_ << vec.transpose().format(CommaInitFmt);
  }

  void write(const std::string& key, const Eigen::RowVectorXd& vec) {
    write_sep();
    write_key(key);
    Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "", "", "", "[", "]");
    *output_ << vec.format(CommaInitFmt);
  }

  /**
   * Reset state
   */
  void reset() { is_rest_ = false; }

private:
  /**
   * Determines whether or not output requires comma separator
   */
  void write_sep() {
    if (is_rest_) {
      *output_ << ", ";
    } else {
      is_rest_ = true;
    }
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
