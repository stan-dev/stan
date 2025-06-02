#ifndef STAN_RUN_CONFIG_DEFAULTS_HPP
#define STAN_RUN_CONFIG_DEFAULTS_HPP

#include <stan/run/config.hpp>
#include <stan/run/metric_type.hpp>

namespace stan {
namespace run {

/**
 * Random seed for initialization.
 */
class random_seed_config : public config<unsigned int> {
private:
  static const std::string description_;
  static void validator(const unsigned int& value) {
    // All unsigned int values are valid.
  }

public:
  random_seed_config() : config<unsigned int>(1, description_, validator) {}

  explicit random_seed_config(unsigned int value) : config<unsigned int>(value, description_) {}

  static unsigned int default_value() { return 1; }
};

const std::string random_seed_config::description_ = "Random seed for initialization.";

/**
 * Default number of chains configuration.
 */
class num_chains_config : public config<size_t> {
private:
  static const std::string description_;
  static void validator(const size_t& value) {
    if (value < 1) {
      throw std::invalid_argument("num_chains must be at least 1");
    }
  }

public:
  num_chains_config() : config<size_t>(1, description_, validator) {}
  
  explicit num_chains_config(size_t value) : config<size_t>(value, description_, validator) {}

  static size_t default_value() { return 1; }
};

const std::string num_chains_config::description_ = "Number of inference chains to run.";

/**
 * Default radius for model parameter initializations.
 */
class init_radius_config : public config<double> {
private:
  static const std::string description_;
  static void validator(const double& value) {
    if (!(value > 0)) {
      throw std::invalid_argument("init_radius must be greater than zero");
    }
  }

public:
  init_radius_config() : config<double>(2.0, description_, validator) {}
  
  explicit init_radius_config(double value) : config<double>(value, description_, validator) {}

  static double default_value() { return 2.0; }
};

const std::string init_radius_config::description_ =
  "Parameter initialization radius, uniform on interval (-init_radius, init_radius).";


/**
 * Number of warmup iterations.
 */
class num_warmup : public config<int> {
private:
  static const std::string description_;
  static void validator(const int& value) {
    if (!(value >= 0)) {
      throw std::invalid_argument("num_warmup must be greater than or equal to 0.");
    }
  }

public:
  num_warmup() : config<int>(1000, description_, validator) {}
  
  explicit num_warmup(int value) : config<int>(value, description_, validator) {}

  static int default_value() { return 1000; }
};

const std::string num_warmup::description_ = "Number of warmup iterations.";

/**
 * Number of sampling iterations.
 */
class num_samples : public config<int> {
private:
  static const std::string description_;
  static void validator(const int& value) {
    if (!(value >= 0)) {
      throw std::invalid_argument("num_samples must be greater than or equal to 0.");
    }
  }

public:
  num_samples() : config<int>(1000, description_, validator) {}
  
  explicit num_samples(int value) : config<int>(value, description_, validator) {}

  static int default_value() { return 1000; }
};

const std::string num_samples::description_ = "Number of sampling iterations.";

/**
 * Period between saved samples.
 */
class thin : public config<int> {
private:
  static const std::string description_;
  static void validator(const int& value) {
    if (!(value > 0)) {
      throw std::invalid_argument("thin must be greater than 0.");
    }
  }

public:
  thin() : config<int>(1, description_, validator) {}
  
  explicit thin(int value) : config<int>(value, description_, validator) {}

  static int default_value() { return 1; }
};

const std::string thin::description_ = "Period between saved samples.";

/**
 * Period between status output messages.
 */
class refresh : public config<int> {
private:
  static const std::string description_;
  // No validator - any refresh value is valid, even negative

public:
  refresh() : config<int>(100, description_) {}
  
  explicit refresh(int value) : config<int>(value, description_) {}

  static int default_value() { return 100; }
};

const std::string refresh::description_ = "Period between status output messages.";
  
/**
 * Step size for discrete evolution.
 */
class stepsize : public config<double> {
private:
  static const std::string description_;
  static void validator(const double& value) {
    if (!(value > 0)) {
      throw std::invalid_argument("stepsize must be greater than 0.");
    }
  }

public:
  stepsize() : config<double>(1.0, description_, validator) {}

  explicit stepsize(double value) : config<double>(value, description_, validator) {}

  static double default_value() { return 1.0; }
};

const std::string stepsize::description_ = "Step size for discrete evolution.";

/**
 * Uniformly random jitter of the stepsize, in percent.
 */
class stepsize_jitter : public config<double> {
private:
  static const std::string description_;
  static void validator(const double& value) {
    if (!(value >= 0 && value <= 1)) {
      throw std::invalid_argument("stepsize_jitter must be between 0 and 1.");
    }
  }

public:
  stepsize_jitter() : config<double>(0.0, description_, validator) {}

  explicit stepsize_jitter(double value) : config<double>(value, description_, validator) {}

  static double default_value() { return 0.0; }
};

const std::string stepsize_jitter::description_ = "Uniformly random jitter of the stepsize, in percent.";

/**
 * Maximum tree depth.
 */
class max_depth : public config<int> {
private:
  static const std::string description_;
  static void validator(const int& value) {
    if (!(value > 0)) {
      throw std::invalid_argument("max_depth must be greater than 0.");
    }
  }

public:
  max_depth() : config<int>(10, description_, validator) {}

  explicit max_depth(int value) : config<int>(value, description_, validator) {}

  static int default_value() { return 10; }
};

const std::string max_depth::description_ = "Maximum tree depth.";

/**
 * Metric type for the Hamiltonian.
 */
class metric_type_config : public config<metric_t> {
private:
  static const std::string description_;
  static void validator(const metric_t& value) {
    // All values in the enum are valid, so no validation needed
  }

public:
  metric_type_config() : config<metric_t>(metric_t::DIAG_E, description_, validator) {}

  explicit metric_type_config(metric_t value) : config<metric_t>(value, description_, validator) {}

  static metric_t default_value() { return metric_t::DIAG_E; }
};

const std::string metric_type_config::description_ = "Type of metric to use in Hamiltonian dynamics.";

/**
 * Adaptation target acceptance statistic.
 */
class delta : public config<double> {
private:
  static const std::string description_;
  static void validator(const double& value) {
    if (!(value > 0 && value < 1)) {
      throw std::invalid_argument("delta must be between 0 and 1 (exclusive).");
    }
  }

public:
  delta() : config<double>(0.8, description_, validator) {}
  
  explicit delta(double value) : config<double>(value, description_, validator) {}

  static double default_value() { return 0.8; }
};

const std::string delta::description_ = "Adaptation target acceptance statistic.";

/**
 * Adaptation regularization scale.
 */
class gamma : public config<double> {
private:
  static const std::string description_;
  static void validator(const double& value) {
    if (!(value > 0)) {
      throw std::invalid_argument("gamma must be greater than 0.");
    }
  }

public:
  gamma() : config<double>(0.05, description_, validator) {}
  
  explicit gamma(double value) : config<double>(value, description_, validator) {}

  static double default_value() { return 0.05; }
};

const std::string gamma::description_ = "Adaptation regularization scale.";

/**
 * Adaptation relaxation exponent.
 */
class kappa : public config<double> {
private:
  static const std::string description_;
  static void validator(const double& value) {
    if (!(value > 0)) {
      throw std::invalid_argument("kappa must be greater than 0.");
    }
  }

public:
  kappa() : config<double>(0.75, description_, validator) {}
  
  explicit kappa(double value) : config<double>(value, description_, validator) {}

  static double default_value() { return 0.75; }
};

const std::string kappa::description_ = "Adaptation relaxation exponent.";

/**
 * Adaptation iteration offset.
 */
class t0 : public config<double> {
private:
  static const std::string description_;
  static void validator(const double& value) {
    if (!(value > 0)) {
      throw std::invalid_argument("t0 must be greater than 0.");
    }
  }

public:
  t0() : config<double>(10.0, description_, validator) {}
  
  explicit t0(double value) : config<double>(value, description_, validator) {}

  static double default_value() { return 10.0; }
};

const std::string t0::description_ = "Adaptation iteration offset.";

/**
 * Width of initial fast adaptation interval.
 */
class init_buffer : public config<unsigned int> {
private:
  static const std::string description_;
  // No validator needed

public:
  init_buffer() : config<unsigned int>(75, description_) {}
  
  explicit init_buffer(unsigned int value) : config<unsigned int>(value, description_) {}

  static unsigned int default_value() { return 75; }
};

const std::string init_buffer::description_ = "Width of initial fast adaptation interval.";

/**
 * Width of final fast adaptation interval.
 */
class term_buffer : public config<unsigned int> {
private:
  static const std::string description_;
  // No validator needed

public:
  term_buffer() : config<unsigned int>(50, description_) {}
  
  explicit term_buffer(unsigned int value) : config<unsigned int>(value, description_) {}

  static unsigned int default_value() { return 50; }
};

const std::string term_buffer::description_ = "Width of final fast adaptation interval.";

/**
 * Initial width of slow adaptation interval.
 */
class window : public config<unsigned int> {
private:
  static const std::string description_;
  // No validator needed

public:
  window() : config<unsigned int>(25, description_) {}
  
  explicit window(unsigned int value) : config<unsigned int>(value, description_) {}

  static unsigned int default_value() { return 25; }
};

const std::string window::description_ = "Initial width of slow adaptation interval.";

}  // namespace run
}  // namespace stan
#endif
