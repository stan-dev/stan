#ifndef STAN_SERVICES_CONFIG_HPP
#define STAN_SERVICES_CONFIG_HPP

#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <stdint.h>

template <typename T>
void error_msg(std::ostream& o, const std::string& name, T val,
               const std::string& error_msg) {
  o << name << "=" << val << ": " << error_msg << std::endl;
}

template <typename T>
inline void print_config(std::ostream& o, std::string key, const T& val) {
  o << "-" << key << " "  << val << std::endl;
}

inline void print_config(std::ostream& o, const std::string& key,
                         const std::string& val) {
  o << "-" << key << " " << '"' << val << '"' << std::endl;
}

// just syntactic sugar for syntactically complex call
inline uint64_t ms_since_epoch() {
  using std::chrono::duration_cast;
  using std::chrono::milliseconds;
  using std::chrono::system_clock;
  return duration_cast<milliseconds>(system_clock::now().time_since_epoch())
      .count();
}

template <typename T>
bool is_positive(std::ostream& o, T x, const std::string& name) {
  bool is_positive = x > 0;
  if (!is_positive)
    error_msg(o, name, x, "value must be positive");
  return is_positive;
}

bool is_within_uint32(std::ostream& o, int64_t x, const std::string& name) {
  bool is_within = x >= 0 && x < std::numeric_limits<uint32_t>::max();
  if (!is_within)
    error_msg(o, name, x, "value must be greater than or equal to 0"
              " and less than 2^32 - 1");
  return is_within;
}

struct model_config {
  std::string model_;
  void set_model(const std::string& model) {
    model_ = model;
  }
  bool validate(std::ostream& o) {
    std::ifstream f(model_);
    bool readable = f.good();
    if (!readable)
      error_msg(o, "model", model_,
                "value must be a path to a readable file.");
    return readable;
  }
  void print(std::ostream& o) {
    print_config(o, "model", model_);
  }
};

struct data_config {
  std::string data_;
  void set_data(const std::string& data) {
    data_ = data;
  }
  bool validate(std::ostream& o) {
    if (data_ == std::string("")) return true;
    std::ifstream f(data_);
    bool readable = f.good();
    if (!readable)
      error_msg(o, "data", data_,
                "value must be a path to a readable file.");
    return readable;
  }
  void print(std::ostream& o) {
    print_config(o, "data", data_);
  }
};

struct rng_seed_config {
  int64_t rng_seed_;
  rng_seed_config()
      : rng_seed_(ms_since_epoch()
                  % std::numeric_limits<uint32_t>::max()) { }
  void set_rng_seed(int32_t rng_seed) {
    rng_seed_ = rng_seed;
  }
  bool validate(std::ostream& o) {
    return is_within_uint32(o, rng_seed_, "rng_seed");
  }
  void print(std::ostream& o) {
    print_config(o, "rng_seed", rng_seed_);
  }
};

struct chain_id_config {
  int64_t chain_id_;
  chain_id_config() : chain_id_(0) { }
  void set_chain_id(int32_t chain_id) {
    chain_id_ = chain_id;
  }
  bool validate(std::ostream& o) {
    return is_within_uint32(o, chain_id_, "chain_id");
  }
  void print(std::ostream& o) {
    print_config(o, "chain_id", chain_id_);
  }
};

struct nuts_config : public model_config, public data_config,
                     public rng_seed_config, public chain_id_config {
  bool validate(std::ostream& o) {
    return this -> model_config::validate(o)
        && this -> data_config::validate(o)
        && this -> rng_seed_config::validate(o)
        && this -> chain_id_config::validate(o);
  }
  void print(std::ostream& o) {
    this -> model_config::print(o);
    this -> data_config::print(o);
    this -> rng_seed_config::print(o);
    this -> chain_id_config::print(o);
  }
};

#endif
