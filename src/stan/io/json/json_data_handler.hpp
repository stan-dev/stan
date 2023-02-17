#ifndef STAN_IO_JSON_JSON_DATA_HANDLER_HPP
#define STAN_IO_JSON_JSON_DATA_HANDLER_HPP

#include <stan/io/json/json_error.hpp>
#include <stan/io/json/json_handler.hpp>
#include <stan/io/json/rapidjson_parser.hpp>
#include <stan/io/var_context.hpp>
#include <cctype>
#include <iostream>
#include <ostream>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <boost/algorithm/string.hpp>

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

struct meta_event {
  // info needed to manage key stack
  enum {
    OBJ_OPEN= 0,  // {
    OBJ_CLOSE = 1,  // }
    KEY = 2, 
  };
};

class array_dims {
  // accumulates array dimensions
 public:
  std::vector<size_t> dims;
  std::vector<size_t> dims_acc;
  int cur_dim;
  array_dims() : dims(), dims_acc(), cur_dim(0) {
  }

  std::string print() {
    std::stringstream ss;
    ss << " num dims: " << dims.size() << "\tdims: ";
    for (size_t i=0; i < dims.size(); ++i)
      ss << " " << dims[i];
    ss << "\tdims_acc: ";
    for (size_t i=0; i < dims_acc.size(); ++i)
      ss << " " << dims_acc[i];
    ss << std::endl;
    return ss.str();
  }

  bool operator==(const array_dims& other) {
    return dims == other.dims && dims_acc == other.dims_acc
        && cur_dim == other.cur_dim;
  }

  bool operator!=(const array_dims& other) {
    return dims != other.dims || dims_acc != other.dims_acc
        || cur_dim != other.cur_dim;
  }

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
  std::map<std::string, array_dims> key_dims_map;
  std::vector<double> values_r;
  std::vector<int> values_i;
  size_t array_start_i;  // index into values_i
  size_t array_start_r;  // index into values_r
  int var_type;
  bool is_int;
  int event;
  
  void dump_state(std::string where) {
    std::string slot_type("unknown");
    if (key_types_map.count(key_str()) == 1)
      slot_type = std::to_string(key_types_map[key_str()]);
    std::cout << where 
              << " key " << key_str() << " var_type " << var_type
              << " slot_type " << slot_type << " is_int " << is_int 
              << "\n\tvalues_i " << values_i.size();
    for (auto& x: values_i)
      std::cout << " " << x;
    std::cout << std::endl;
    std::cout<< "\n\tvalues_r " << values_r.size();
    for (auto& x: values_r)
      std::cout << " " << x;
    std::cout << std::endl;
    if (key_dims_map.count(key_str()) == 1)
      std::cout << key_dims_map[key_str()].print();
    else
      std::cout << std::endl;
  }

  void reset_values() {
    values_r.clear();
    values_i.clear();
    array_start_i = 0;
    array_start_r = 0;
  }

  void reset_var() {
    reset_values();
    var_type = meta_type::SCALAR;
    is_int = true;
  }

  bool is_init() {
    return (key_stack.empty() && var_types_map.empty()  && key_types_map.empty()
            && values_r.empty() && values_i.empty() && key_dims_map.empty()
            && array_start_i == 0 && array_start_r == 0 && is_int);
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
        key_dims_map(),
        values_r(),
        values_i(),
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
    for (auto& x : key_types_map)
      std::cout << "key " << x.first << " type " << x.second << std::endl;
    for (auto& x : var_types_map)
      std::cout << " variable " << x.first << " type " << x.second << std::endl;
    for (auto& x : key_dims_map) {
      std::cout << " variable " << x.first;
      std::cout << x.second.print();
    }
    std::cout << std::endl;
    convert_arrays();
    for (auto& var : vars_i)
      std::cout << var.first << std::endl;
    for (auto& var : vars_r)
      std::cout << var.first << std::endl;

    //    reset_var();
  }

  void key(const std::string &key) {
    if (event != meta_event::OBJ_OPEN) {
      save_key_value_pair();
    }
    event = meta_event::KEY;
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
    event = meta_event::OBJ_OPEN;
    if (is_init())
      return;
    if (var_type == meta_type::SCALAR)
      var_type = meta_type::TUPLE;
    else if (var_type == meta_type::ARRAY)
      var_type = meta_type::ARRAY_OF_TUPLES;
    key_types_map[key_str()] = var_type;
  }

  void end_object() {
    event = meta_event::OBJ_CLOSE;
    if (var_type == meta_type::ARRAY_OF_TUPLES) {
      array_dims outer = get_outer_dims(key_stack);
      if (!outer.dims.empty()) {
        outer.dims_acc[outer.dims.size()-1]++;
        set_outer_dims(outer);
      }
    }
    save_key_value_pair();
  }

  void start_array() {
    if (key_stack.empty()) {
      throw json_error("expecting JSON object, found array");
    }
    if (var_type == meta_type::SCALAR
        && !(values_r.empty() && values_r.empty())) {
      std::stringstream errorMsg;
      errorMsg << "variable: " << key_str() << ", error: non-scalar array value";
      throw json_error(errorMsg.str());
    }
    if (var_type == meta_type::SCALAR)
      var_type = meta_type::ARRAY;
    key_types_map[key_str()] = meta_type::ARRAY;
    array_dims dims;
    if (key_dims_map.count(key_str()) == 1)
      dims = key_dims_map[key_str()];
    dims.cur_dim++;
    if (dims.dims.empty() || dims.dims.size() < dims.cur_dim) {
      dims.dims.push_back(0);
      dims.dims_acc.push_back(0);
    }
    if (dims.cur_dim > 1)
      dims.dims_acc[dims.cur_dim-2]++;
    key_dims_map[key_str()] = dims;
    array_start_i = values_i.size();
    array_start_r = values_r.size();
  }

  void end_array() {
    if (key_dims_map.count(key_str()) == 0)
      unexpected_error(key_str());
    array_dims dims = key_dims_map[key_str()];
    bool is_aot = false;
    if (var_type == meta_type::ARRAY_OF_TUPLES) {
      array_dims outer = get_outer_dims(key_stack);
      if (outer == dims)
        is_aot = true;
    }
    bool is_last = !is_aot && dims.cur_dim == dims.dims.size();
    int idx = dims.cur_dim - 1;
    if (is_last && 0 == dims.dims[idx]) {  // innermost row of scalar elts
      if (is_int)
        dims.dims[idx] = values_i.size() - array_start_i;
      else
        dims.dims[idx] = values_r.size() - array_start_r;
    } else if (0 == dims.dims[idx]) {  // row of array or tuple elts
      dims.dims[idx] = dims.dims_acc[idx];
    } else {
      bool is_rect = false;
      if (is_last) {
        if ((is_int && dims.dims[idx] == values_i.size() - array_start_i)
            || (!is_int && dims.dims[idx] == values_r.size() - array_start_r))
          is_rect = true;
      } else if (dims.dims[idx] == dims.dims_acc[idx]) {
        is_rect = true;
      }
      if (!is_rect) {
        std::stringstream errorMsg;
        errorMsg << "variable: " << key_str() << ", error: non-rectangular array";
        throw json_error(errorMsg.str());
      }
    }
    dims.dims_acc[idx] = 0;
    dims.cur_dim--;
    key_dims_map[key_str()] = dims;
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
    if (key_types_map[key_str()] == meta_type::TUPLE
        || key_types_map[key_str()] == meta_type::ARRAY_OF_TUPLES) {
      ;
    } else {
      dump_state("save scalar or array (row)");
      std::vector<size_t> dims;
      if (key_dims_map.count(key_str()) == 1)
        dims = key_dims_map[key_str()].dims;
      bool is_new = vars_r.count(key_str()) == 0 && vars_i.count(key_str()) == 0;
      bool is_real = vars_r.count(key_str()) == 1;
      bool was_int = vars_i.count(key_str()) == 1;
      std::cout << "is_new? " << is_new << " is_real? " << is_real << " was_int? " << was_int << std::endl;
      if (is_new) {
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
      } else {
        if (var_type != meta_type::ARRAY_OF_TUPLES) {
          std::stringstream errorMsg;
          errorMsg << "attempt to redefine variable " << key_str();
          throw json_error(errorMsg.str());
        }
        std::cout << "one row of array of tuples" << std::endl;
        var_types_map[key_str()] = meta_type::ARRAY;
        std::vector<size_t> dims = key_dims_map[key_str()].dims;
        if ((!is_int && was_int) || (is_int && is_real)) {  // promote to double
          std::vector<double> values_tmp;
          std::cout << "save existing vars_i values ";
          for (auto& x : vars_i[key_str()].first) {
            values_tmp.push_back(x);
            std::cout << " " << x;
          }
          std::cout << std::endl;
          for (auto&x : values_r)
            values_tmp.push_back(x);
          std::pair<std::vector<double>, std::vector<size_t>> pair;
          pair = make_pair(values_tmp, dims);
          vars_r[key_str()] = pair;
          vars_i.erase(key_str());
        } else if (is_int) {
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
      }
    }
    key_stack.pop_back();
  }

  void convert_arrays() {
    for (auto const &var : var_types_map) {
      if (var.second != meta_type::ARRAY) {
        std::cout << "nothing to be done for " << var.first << std::endl;
        return;
      }
      std::cout << "converting " << var.first << " type " << var.second << std::endl;
      std::vector<size_t> all_dims;
      array_dims inner = key_dims_map[var.first];
      // check is need to combine dims
      std::vector<std::string> keys;
      split(keys, var.first, boost::is_any_of("."), boost::token_compress_on);
      array_dims outer = get_outer_dims(keys);
      if (inner != outer)
        for (auto& x : outer.dims)
          all_dims.push_back(x);
      for (auto& x : inner.dims)
        all_dims.push_back(x);
      if (vars_i.count(var.first) == 1) {
        std::vector<int> cm_values_i(vars_i[var.first].first.size());
        std::pair<std::vector<int>, std::vector<size_t>> pair;
        if (all_dims.empty()) {
          to_column_major(cm_values_i,
                          vars_i[var.first].first,
                          vars_i[var.first].second);
          pair = make_pair(cm_values_i, vars_i[var.first].second);
        } else {
          to_column_major(cm_values_i,
                          vars_i[var.first].first,
                          all_dims);
          pair = make_pair(cm_values_i, all_dims);
        }            
        vars_i[var.first] = pair;
      } else if (vars_r.count(var.first) == 1) {
        std::vector<double> cm_values_r(vars_r[var.first].first.size());
        std::pair<std::vector<double>, std::vector<size_t>> pair;
        if (all_dims.empty()) {
          to_column_major(cm_values_r,
                          vars_r[var.first].first,
                          vars_r[var.first].second);
          pair = make_pair(cm_values_r, vars_r[var.first].second);
        } else {
          to_column_major(cm_values_r,
                          vars_r[var.first].first,
                          all_dims);
          pair = make_pair(cm_values_r, all_dims);
        }            
        vars_r[var.first] = pair;
      } else {
        unexpected_error("cannot convert " + var.first);
      }
    }
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

  void unexpected_error(std::string where) {
    std::stringstream errorMsg;
    errorMsg << "json_data_handler unexpected parsing error, at key " << where;
    throw json_error(errorMsg.str());
  }    

  // assumes at more 2 dims - inner / outer
  array_dims get_outer_dims(std::vector<std::string> keys) {
    std::string prefix;
    for (size_t i=0; i < keys.size(); ++i) {
      prefix.append(keys[i]);
      if (key_dims_map.count(prefix) == 1)
        return key_dims_map[prefix];
      else
        prefix.append(".");
    }
    array_dims empty;
    return empty;
  }

  void set_outer_dims(array_dims update) {
    std::string prefix;
    for (size_t i=0; i < key_stack.size(); ++i) {
      prefix.append(key_stack[i]);
      if (key_dims_map.count(prefix) == 1)
        break;
      else
        prefix.append(".");
    }
    if (prefix.back() != '.')
      key_dims_map[prefix] = update;
    else
      unexpected_error(key_str());
  }

};

}  // namespace json

}  // namespace stan

#endif
