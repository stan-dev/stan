#ifndef TEST_UNIT_SERVICES_PATHFINDER_UTIL_HPP
#define TEST_UNIT_SERVICES_PATHFINDER_UTIL_HPP

struct mock_callback : public stan::callbacks::interrupt {
  int n;
  mock_callback() : n(0) {}

  void operator()() { n++; }
};

class loggy : public stan::callbacks::logger {
  /**
   * Logs a message with debug log level
   *
   * @param[in] message message
   */
  virtual void debug(const std::string& message) {
    std::cout << message << "\n";
  }

  /**
   * Logs a message with debug log level.
   *
   * @param[in] message message
   */
  virtual void debug(const std::stringstream& message) {
    std::cout << message.str() << "\n";
  }

  /**
   * Logs a message with info log level.
   *
   * @param[in] message message
   */
  virtual void info(const std::string& message) {
    std::cout << message << "\n";
  }

  /**
   * Logs a message with info log level.
   *
   * @param[in] message message
   */
  virtual void info(const std::stringstream& message) {
    std::cout << message.str() << "\n";
  }

  /**
   * Logs a message with warn log level.
   *
   * @param[in] message message
   */
  virtual void warn(const std::string& message) {
    std::cout << message << "\n";
  }

  /**
   * Logs a message with warn log level.
   *
   * @param[in] message message
   */
  virtual void warn(const std::stringstream& message) {
    std::cout << message.str() << "\n";
  }

  /**
   * Logs an error with error log level.
   *
   * @param[in] message message
   */
  virtual void error(const std::string& message) {
    std::cout << message << "\n";
  }

  /**
   * Logs an error with error log level.
   *
   * @param[in] message message
   */
  virtual void error(const std::stringstream& message) {
    std::cout << message.str() << "\n";
  }

  /**
   * Logs an error with fatal log level.
   *
   * @param[in] message message
   */
  virtual void fatal(const std::string& message) {
    std::cout << message << "\n";
  }

  /**
   * Logs an error with fatal log level.
   *
   * @param[in] message message
   */
  virtual void fatal(const std::stringstream& message) {
    std::cout << message.str() << "\n";
  }
};

class values : public stan::callbacks::stream_writer {
 public:
  std::vector<std::string> names_;
  std::vector<std::string> times_;
  std::vector<std::vector<double>> states_;
  std::vector<Eigen::VectorXd> eigen_states_;
  std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd>> optim_path_;
  Eigen::MatrixXd values_;
  values(std::ostream& stream) : stan::callbacks::stream_writer(stream) {}

  /**
   * Writes a set of names.
   *
   * @param[in] names Names in a std::vector
   */
  void operator()(const std::vector<std::string>& names) { names_ = names; }

  void operator()(const std::string& times) { times_.push_back(times); }
  void operator()() { times_.push_back("\n"); }
  /**
   * Writes a set of values.
   *
   * @param[in] state Values in a std::vector
   */
  void operator()(const std::vector<double>& state) {
    states_.push_back(state);
  }
  void operator()(
      const std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd>>& xx) {
    optim_path_ = xx;
  }
  void operator()(
      const std::tuple<Eigen::VectorXd, Eigen::VectorXd>& xx) {
    optim_path_.push_back(xx);
  }
  template <typename EigVec, stan::require_eigen_vector_t<EigVec>* = nullptr>
  void operator()(const EigVec& vals) {
    eigen_states_.push_back(vals);
  }
  template <typename EigMat,
            stan::require_eigen_matrix_dynamic_t<EigMat>* = nullptr>
  void operator()(const EigMat& vals) {
    values_ = vals;
  }
};

#endif
