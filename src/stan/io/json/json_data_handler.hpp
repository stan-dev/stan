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
  std::vector<size_t> dim_elts;  // accumulates elements per row
  int dim_at;
  int var_type;
  size_t array_start_i;  // index into values_i
  size_t array_start_r;  // index into values_r
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
                << " dims " << dims.size() << " dim_at " << dim_at
                << " values_i " << values_i.size() << " values_r " << values_r.size()
                << std::endl;
      std::cout << "\tdims: ";
      for (size_t i = 0; i < dims.size(); i++)
        std::cout << " " << dims[i];
      std::cout << "\tdim_elts: ";
      for (size_t i = 0; i < dim_elts.size(); i++)
        std::cout << " " << dim_elts[i];
      std::cout << std::endl;
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
    dim_elts.clear();
    dim_at = 0;
    var_type = meta_type::SCALAR;
    is_int = true;
  }

  bool is_init() {
    return (key_stack.empty() && var_types_map.empty()  && key_types_map.empty()
            && values_r.empty() && values_i.empty() && dims.empty()
            && dim_elts.empty() && dim_at == 0 && array_start_i == 0
            && array_start_r == 0 && is_int);
  }

  std::string key_str() {
    if (key_stack.empty()) return "";
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
        dim_elts(),
        dim_at(0),
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
    reset_var();
  }

  void key(const std::string &key) {
    save_key_value_pair();
    if (key_stack.empty()) {
      reset_var();
    } else {
      reset_values();
    }
    key_stack.push_back(key);
    if (key_types_map.count(key_str()) == 0) {
      key_types_map[key_str()] = meta_type::SCALAR;
      is_int = true;
    }
  }

  void start_object() {
    if (is_init())
      return;
    key_types_map[key_str()] = meta_type::TUPLE;
    if (var_type == meta_type::SCALAR)
      var_type = meta_type::TUPLE;
    else if (var_type == meta_type::ARRAY)
      var_type = meta_type::ARRAY_OF_TUPLES;
  }

  void end_object() {
    save_key_value_pair();
    if (key_stack.size() > 0 && var_type != meta_type::ARRAY_OF_TUPLES)
      key_stack.pop_back();
  }

  void start_array() {
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
    dim_at++;
    if (dims.empty() || dims.size() < dim_at) {
      dims.push_back(0);
      dim_elts.push_back(0);
    }
    if (dim_at > 1)
      dim_elts[dim_at-2]++;
    array_start_i = values_i.size();
    array_start_r = values_r.size();
  }

  void end_array() {
    if (dim_at == dims.size() && 0 == dims[dim_at-1]) {
      // first inner array row 
      if (is_int)
        dims[dim_at-1] = values_i.size() - array_start_i;
      else
        dims[dim_at-1] = values_r.size() - array_start_r;
    } else if (dim_at < dims.size() && 0 == dims[dim_at-1]) {
      // first outer array row
      dims[dim_at-1] = dim_elts[dim_at-1];
    } else if (dims[dim_at-1] > 0) {
      bool is_rect = false;
      if (dim_at == dims.size()) {
        if ((is_int && dims[dim_at-1] == values_i.size() - array_start_i)
            || (!is_int && dims[dim_at-1] == values_r.size() - array_start_r))
          is_rect = true;
      } else if (dims[dim_at-1] == dim_elts[dim_at-1]) {
        is_rect = true;
      }
      if (!is_rect) {
        std::stringstream errorMsg;
        errorMsg << "variable: " << key_str() << ", error: non-rectangular array";
        throw json_error(errorMsg.str());
      }
    }
    dim_elts[dim_at-1] = 0;
    dim_at--;
    if (dim_at == 0) {
      save_key_value_pair();
      if (key_stack.size() > 0)
        key_stack.pop_back();
    }
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
  }

  void number_int64(int64_t n) {
    // the number doesn't fit in int (otherwise number_int() would be called)
    number_double(n);
  }

  void number_unsigned_int64(uint64_t n) {
    // the number doesn't fit in int (otherwise number_unsigned_int() would be
    // called)
    number_double(n);
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
    if (0 == key_stack.size())
      return;
    if (key_types_map.count(key_str()) < 1)
      unexpected_error(key_str());
    if (key_types_map[key_str()] == meta_type::TUPLE)
      return;

    if (var_type == meta_type::ARRAY_OF_TUPLES) {
      var_types_map[key_str()] = meta_type::ARRAY;
      // update sizes - 1-d array, incr last dim, 2-d array, incr last 2 dims
      if (is_int) {
        if (vars_r.count(key_str()) == 0 && vars_i.count(key_str()) == 0) {
          std::pair<std::vector<int>, std::vector<size_t>> pair;
          pair = make_pair(values_i, dims);
          vars_i[key_str()] = pair;
        } else if (vars_r.count(key_str()) == 0) {
          for (std::vector<int>::iterator it = values_i.begin();
               it != values_i.end(); ++it)
            vars_i[key_str()].first.push_back(*it);
          vars_i[key_str()].second = dims;
        } else {
          for (std::vector<double>::iterator it = values_r.begin();
               it != values_r.end(); ++it)
            vars_r[key_str()].first.push_back(*it);
          vars_r[key_str()].second = dims;
        }          
      } else {
        if (vars_i.count(key_str()) == 1) {
          std::vector<double> tmp(vars_i[key_str()].first.begin(), vars_i[key_str()].first.end());
          for (std::vector<double>::iterator it = values_r.begin();
               it != values_r.end(); ++it)
            tmp.push_back(*it);
          std::pair<std::vector<double>, std::vector<size_t>> pair;
          pair = make_pair(tmp, dims);
          vars_r[key_str()] = pair;
          vars_i.erase(key_str());
        } else if (vars_r.count(key_str()) == 0) {
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
    } else {
      var_types_map[key_str()] = key_types_map[key_str()];
      if (is_int) {
        std::pair<std::vector<int>, std::vector<size_t>> pair;
        pair = make_pair(values_i, dims);
        vars_i[key_str()] = pair;
      } else {
        std::pair<std::vector<double>, std::vector<size_t>> pair;
        pair = make_pair(values_r, dims);
        vars_r[key_str()] = pair;
      }
    }
    key_stack.pop_back();
  }

  void convert_arrays() {
    for (auto const &var : var_types_map)
      std::cout << var.first << " " << var.second << " ";
    std::cout << std::endl;
    // for (auto const &var : var_types_map) {
    //   if (var.second == meta_type::ARRAY) {
    //     std::cout << "converting " << var.first << std::endl;
    //     if (vars_i.count(var.first) == 1) {
    //       std::pair<std::vector<int>, std::vector<size_t>> pair;
    //       std::vector<int> cm_values_i(vars_i[var.first].first.size());
    //       to_column_major(cm_values_i,
    //                       vars_i[var.first].first,
    //                       vars_i[var.first].second);
    //       pair = make_pair(cm_values_i, vars_i[var.first].second);
    //       vars_i[var.first] = pair;
    //     } else if (vars_r.count(var.first) == 1) {
    //       std::pair<std::vector<double>, std::vector<size_t>> pair;
    //       std::vector<double> cm_values_r(vars_r[var.first].first.size());
    //       to_column_major(cm_values_r,
    //                       vars_r[var.first].first,
    //                       vars_r[var.first].second);
    //       pair = make_pair(cm_values_r, vars_r[var.first].second);
    //       vars_r[var.first] = pair;

    //     } else {
    //       unexpected_error(var.first);
    //     }
    //   }
    // }
  }
  
  template <typename T>
  void to_column_major(std::vector<T> &cm_vals, const std::vector<T> &rm_vals,
                       const std::vector<size_t> &dims) {
    std::cout << "to_column_major, num_dims: " << dims.size() << ", sizes: ";
    for (size_t i = 0; i < dims.size(); i++)
      std::cout << " " << dims[i];
    std::cout << " num elts " << rm_vals.size() << std::endl;

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

  void unexpected_error(std::string where) {
    std::stringstream errorMsg;
    errorMsg << "json_data_handler unexpected parsing error, at key " << where;
    throw json_error(errorMsg.str());
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
