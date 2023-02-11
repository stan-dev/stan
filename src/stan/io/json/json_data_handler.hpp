#ifndef STAN_IO_JSON_JSON_DATA_HANDLER_HPP
#define STAN_IO_JSON_JSON_DATA_HANDLER_HPP

#include <stan/io/json/json_error.hpp>
#include <stan/io/json/json_handler.hpp>
#include <stan/io/json/rapidjson_parser.hpp>
#include <stan/io/var_context.hpp>
#include <cctype>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <boost/algorithm/string/predicate.hpp>

namespace stan {

namespace json {

typedef std::pair<std::vector<double>, std::vector<size_t>> var_r;
typedef std::pair<std::vector<int>, std::vector<size_t>> var_i;

typedef std::map<std::string, var_r> vars_map_r;
typedef std::map<std::string, var_i> vars_map_i;

struct meta_type {
  // the general set of structures that the handler needs to manage
  // as determined by the initial sequence of start elements
  // following the Stan variable name,
  enum {
    SCALAR = 0,  // no start elements
    ARRAY = 1,   // one or more "["
    TUPLE = 2,   // one or more "{"
    ARRAY_OF_TUPLES = 3,  // one or more "[" followed by "{"
  };
};

/**
 * A <code>json_data_handler</code> is an implementation of a
 * <code>json_handler</code> that restricts the allowed JSON text
 * to a set of Stan variable declarations in JSON format.
 * Each Stan variable consists of a JSON key : value pair.
 * The key is a string (the Stan variable name) and the value
 * is either a scalar variables, array, or a tuple.
 * The latter two kinds of variables allow for deeply nested
 * structures, e.g., arrays of tuples, tuples composed of arrays,
 * tuples composed of arrays of tuples, etc.
 *
 * <p>The <code>json_data_handler</code> checks that the top-level
 * JSON object contains a set of key-value pairs.
 * The strings \"Inf\" and \"Infinity\" are mapped to positive infinity,
 * the strings \"-Inf\" and \"-Infinity\" are mapped to negative infinity,
 * and the string \"NaN\" is mapped to not-a-number.
 * Bare versions of Infinity, -Infinity, and NaN are also allowed.
 */
class json_data_handler : public stan::json::json_handler {
 private:
  vars_map_r &vars_r;
  vars_map_i &vars_i;
  std::vector<std::string> key_stack;
  std::map<std::string, int> var_types_map;
  std::map<std::string, int> key_types_map;
  std::vector<double> values_r;
  std::vector<int> values_i;
  std::vector<size_t> dims;
  std::vector<size_t> dims_found; // element per dim so far
  std::vector<bool> dims_known; // false until seen first row
  int var_type;
  int dim_idx;  // current dimension
  size_t array_start_i;  // ptr into values_i
  size_t array_start_r;  // ptr into values_r
  bool is_int;

  bool debug = true;

  void dump_state(std::string where) {
    if (debug) {
      std::string slot_type("unknown");
      if (key_types_map.count(key_str()) == 1)
        slot_type = std::to_string(key_types_map[key_str()]);
      std::cout << where << std::endl
                << "key " << key_str() << " var_type " << var_type
                << " slot_type " << slot_type << " is_int " << is_int 
                << " dims " << dims.size() << " dim_idx " << dim_idx
                << " values_i " << values_i.size() << " values_r " << values_r.size()
                << std::endl;
    }
  }

  void reset_values() {
    values_r.clear();
    values_i.clear();
    array_start_i = 0;
    array_start_r = 0;
  }

  void reset_var() {
    reset_values();
    dims.clear();
    dims_found.clear();
    dims_known.clear();
    dim_idx = -1;
    var_type = meta_type::SCALAR;
    is_int = true;
  }

  bool is_init() {
    return (key_stack.size() == 0 && var_types_map.size() == 0
            && key_types_map.size() == 0 && values_r.size() == 0
            && values_i.size() == 0 && dims.size() == 0
            && dims_found.size() == 0 && dims_known.size() == 0
            && dim_idx == -1 && array_start_i == 0 && array_start_r == 0
            && is_int);
  }

  std::string key_str() {
    if (key_stack.size() == 0) return "";
    return std::accumulate(std::next(key_stack.begin()), key_stack.end(),
                           key_stack[0], // start with first element
                           [](std::string a, const std::string b) {
                             return std::move(a) + '.' + b;
                           });
  }

 public:
  /**
   * Construct a json_data_handler object.
   *
   * <b>Warning:</b> This method does not close the input stream.
   *
   * @param a_vars_r name-value map for real-valued variables
   * @param a_vars_i name-value map for int-valued variables
   */
  json_data_handler(vars_map_r &a_vars_r, vars_map_i &a_vars_i)
      : json_handler(),
        vars_r(a_vars_r),
        vars_i(a_vars_i),
        key_stack(),
        var_types_map(),
        key_types_map(),
        values_r(),
        values_i(),
        dims(),
        dim_idx(-1),
        array_start_i(0),
        array_start_r(0),
        var_type(meta_type::SCALAR),
        is_int(true) {}
  
  // *** start handler events ***
  void start_text() {
    vars_i.clear();  // can't accumulate var defs across calls to parser
    vars_r.clear();
    var_types_map.clear();
    key_types_map.clear();
    reset_var();
  }

  void end_text() {
    save_key_value_pair();
    convert_arrays();
    //    reset_var();
  }

  void key(const std::string &key) {
    save_key_value_pair();
    if (key_stack.size() == 0) {
      reset_var();
    } else {
      reset_values();
    }
    key_stack.push_back(key);
    if (key_types_map.count(key_str()) == 0) {
      key_types_map[key_str()] = meta_type::SCALAR;
      is_int = true;
    }
    dump_state("handled key");
  }

  void start_object() {
    if (is_init())
      return;
    key_types_map[key_str()] = meta_type::TUPLE;
    if (var_type == meta_type::SCALAR)
      var_type = meta_type::TUPLE;
    else if (var_type == meta_type::ARRAY)
      var_type = meta_type::ARRAY_OF_TUPLES;
    dump_state("handled {");
  }

  void end_object() {
    save_key_value_pair();
    dump_state("handled }");
    if (key_stack.size() > 0 && dim_idx < 0)
      key_stack.pop_back();
  }

  void start_array() {
    dump_state("start [");
    if (0 == key_stack.size()) {
      throw json_error("expecting JSON object, found array");
    }
    if (var_type == meta_type::SCALAR
        && (values_r.size() > 0 || values_i.size() > 0)) {
      std::stringstream errorMsg;
      errorMsg << "variable: " << key_str() << ", error: non-scalar array value";
      throw json_error(errorMsg.str());
    }
    if (var_type == meta_type::SCALAR)
      var_type = meta_type::ARRAY;
    key_types_map[key_str()] = meta_type::ARRAY;
    dim_idx++;
    if (dims.size() < dim_idx + 1) {
      dims.push_back(0);
      dims_found.push_back(0);
      dims_known.push_back(false);
    }
    array_start_i = values_i.size();
    array_start_r = values_r.size();
    dump_state("handled [");
  }

  void end_array() {
    if (dims_known[dim_idx]) {
      if ((is_int && dims[dim_idx] != values_i.size() - array_start_i)
          || (!is_int && dims[dim_idx] != values_r.size() - array_start_r)) {
          std::stringstream errorMsg;
          errorMsg << "variable: " << key_str() << ", error: non-rectangular array";
          throw json_error(errorMsg.str());
        }
    } else {
      if (is_int)
        dims[dim_idx] = values_i.size() - array_start_i;
      else
        dims[dim_idx] = values_r.size() - array_start_r;
      dims_known[dim_idx] = true;
    }
    dim_idx--;
    if (dim_idx < 0) {
      save_key_value_pair();
      if (key_stack.size() > 0)
        key_stack.pop_back();
    }
    dump_state("handled ]");
  }

  void null() {
    std::stringstream errorMsg;
    errorMsg << "variable: " << key_str() << ", error: null values not allowed";
    throw json_error(errorMsg.str());
  }

  void boolean(bool p) {
    std::stringstream errorMsg;
    errorMsg << "variable: " << key_str() << ", error: boolean values not allowed";
    throw json_error(errorMsg.str());
  }

  void string(const std::string &s) {
    double tmp;
    if (0 == s.compare("-Inf")) {
      tmp = -std::numeric_limits<double>::infinity();
    } else if (0 == s.compare("-Infinity")) {
      tmp = -std::numeric_limits<double>::infinity();
    } else if (0 == s.compare("Inf")) {
      tmp = std::numeric_limits<double>::infinity();
    } else if (0 == s.compare("Infinity")) {
      tmp = std::numeric_limits<double>::infinity();
    } else if (0 == s.compare("NaN")) {
      tmp = std::numeric_limits<double>::quiet_NaN();
    } else {
      std::stringstream errorMsg;
      errorMsg << "variable: " << key_str() << ", error: string values not allowed";
      throw json_error(errorMsg.str());
    }
    promote_to_double();
    values_r.push_back(tmp);
  }

  void number_double(double x) {
    promote_to_double();
    values_r.push_back(x);
  }

  void number_int(int n) {
    if (is_int) {
      values_i.push_back(n);
    } else {
      values_r.push_back(n);
    }
    dump_state("int " + std::to_string(n));
  }

  void number_unsigned_int(unsigned n) {
    // if integer overflow, promote numeric data to double
    if (n > (unsigned)std::numeric_limits<int>::max())
      promote_to_double();
    if (is_int) {
      values_i.push_back(static_cast<int>(n));
    } else {
      values_r.push_back(n);
    }
    dump_state("int " + std::to_string(n));
  }

  void number_int64(int64_t n) {
    // the number doesn't fit in int (otherwise number_int() would be called)
    number_double(n);
    dump_state("int " + std::to_string(n));
  }

  void number_unsigned_int64(uint64_t n) {
    // the number doesn't fit in int (otherwise number_unsigned_int() would be
    // called)
    number_double(n);
    dump_state("int " + std::to_string(n));
  }

  // *** end handler events ***

  void promote_to_double() {
    if (is_int) {
      for (std::vector<int>::iterator it = values_i.begin();
           it != values_i.end(); ++it)
        values_r.push_back(*it);
      array_start_r = array_start_i;
    }
    is_int = false;
  }


  void save_key_value_pair() {
    dump_state("save key");
    if (0 == key_stack.size())
      return;
    if (key_types_map.count(key_str()) < 1) {
      // this shouldn't happen!!!
      std::stringstream errorMsg;
      errorMsg << "json_data_handler unexpected parsing error " << key_str();
      throw json_error(errorMsg.str());
    }
    if (key_types_map[key_str()] == meta_type::TUPLE) {
      std::cout << " tuple var " << key_str() << " nothing to record " << std::endl;
      return;
    }
    bool do_append = (var_type == meta_type::ARRAY_OF_TUPLES);
    std::cout << "value type " << key_types_map[key_str()] << std::endl;
    var_types_map[key_str()] = key_types_map[key_str()];
    if (!do_append) {
      if (is_int) {
        std::pair<std::vector<int>, std::vector<size_t>> pair;
        pair = make_pair(values_i, dims);
        vars_i[key_str()] = pair;
      } else {
        std::pair<std::vector<double>, std::vector<size_t>> pair;
        pair = make_pair(values_r, dims);
        vars_r[key_str()] = pair;
      }
    } else {
        // promote to double now!!!!
      if (is_int) {
        if (vars_i.count(key_str()) == 0) {
          std::pair<std::vector<int>, std::vector<size_t>> pair;
          pair = make_pair(values_i, dims);
          vars_i[key_str()] = pair;
        } else {
          for (std::vector<int>::iterator it = values_i.begin();
               it != values_i.end(); ++it)
            vars_i[key_str()].first.push_back(*it);
          vars_i[key_str()].second = dims;
        }
      } else {
        if (vars_r.count(key_str()) == 0) {
          std::pair<std::vector<double>, std::vector<size_t>> pair;
          pair = make_pair(values_r, dims);
          vars_r[key_str()] = pair;
        } else {
          for (std::vector<double>::iterator it = values_r.begin();
               it != values_r.end(); ++it)
            vars_r[key_str()].first.push_back(*it);
          vars_r[key_str()].second = dims;
        }
      }
    }
    key_stack.pop_back();
  }

  void convert_arrays() {
    for (auto const &var : var_types_map)
      std::cout << var.first << " " << var.second << " ";
    std::cout << std::endl;
  }
  
  template <typename T>
  void to_column_major(std::vector<T> &cm_vals, const std::vector<T> &rm_vals,
                       const std::vector<size_t> &dims) {
    for (size_t i = 0; i < rm_vals.size(); i++) {
      size_t idx = convert_offset_rtl_2_ltr(i, dims);
      cm_vals[idx] = rm_vals[i];
    }
  }

  // convert row-major offset to column-major offset
  size_t convert_offset_rtl_2_ltr(size_t rtl_offset,
                                  const std::vector<size_t> &dims) {
    size_t rtl_dsize = 1;
    for (size_t i = 1; i < dims.size(); i++)
      rtl_dsize *= dims[i];

    // array index should be valid, but check just in case
    if (rtl_offset >= rtl_dsize * dims[0]) {
      std::stringstream errorMsg;
      errorMsg << "variable: " << key_str() << ", unexpected error";
      throw json_error(errorMsg.str());
    }

    // calculate offset by working left-to-right to get array indices
    // for row-major offset left-most dimensions are divided out
    // for column-major offset successive dimensions are multiplied in
    size_t rem = rtl_offset;
    size_t ltr_offset = 0;
    size_t ltr_dsize = 1;
    for (size_t i = 0; i < dims.size() - 1; i++) {
      size_t idx = rem / rtl_dsize;
      ltr_offset += idx * ltr_dsize;
      rem = rem - idx * rtl_dsize;
      rtl_dsize = rtl_dsize / dims[i + 1];
      ltr_dsize *= dims[i];
    }
    ltr_offset += rem * ltr_dsize;  // for loop stops 1 early

    return ltr_offset;
  }
};

}  // namespace json

}  // namespace stan

#endif


    // edge case:  adding to array_of_tuples slot, promote to double
    // // transpose order of array values to column-major
    // if (is_int) {
    //   
    //   if (dims.size() > 1) {
    //     std::vector<int> cm_values_i(values_i.size());
    //     to_column_major(cm_values_i, values_i_, dims);
    //     pair = make_pair(cm_values_i, dims);

    //   } else {
    //     
    //   }
    //   if (do_append && (vars_i.find(key_str()) != vars_i.end()))
    //     // check dims!
    //     for (size_t i = 0; i < vars_i.size(); i++)
    //       vars_i[key_str()].first.push_back(values_i[i]);
    //   else
    //     vars_i[key_str()] = pair;
    //   std::cout << " save int " << key_str() << std::endl;
    // } else {
    //   std::pair<std::vector<double>, std::vector<size_t>> pair;
    //   if (dims.size() > 1) {
    //     std::vector<double> cm_values_r(values_r.size());
    //     to_column_major(cm_values_r, values_r_, dims);
    //     pair = make_pair(cm_values_r, dims);
    //   } else {
    //     pair = make_pair(values_r_, dims);
    //   }
    //   if (do_append && (vars_r.find(key_str()) != vars_r.end()))
    //     // check dims! - dims must match existing dims
    //     for (size_t i = 0; i < vars_r.size(); i++)
    //       vars_r[key_str()].first.push_back(values_r[i]);
    //   else
    //     vars_r[key_str()] = pair;
    //   std::cout << " save real " << key_str() << std::endl;
    // }
