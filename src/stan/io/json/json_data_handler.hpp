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

/** Enum of the kinds of structures the handler needs to manage.
 *  Determined by the initial sequence of start elements following
 *  the top-level set of keys in the JSON object.
 */
struct meta_type {
  enum {
    SCALAR = 0,           // no start elements
    ARRAY = 1,            // one or more "["
    TUPLE = 2,            // one or more "{"
    ARRAY_OF_TUPLES = 3,  // one or more "[" followed by "{"
  };
};

/** Enum for salient handler events */
struct meta_event {
  enum {
    OBJ_OPEN = 0,   // {
    OBJ_CLOSE = 1,  // }
    KEY = 2,
  };
};

/** Tracks array dimensions.
 *  Vector 'dims_acc' records number of elements seen since open_array.
 *  Vector 'dims' records size of first row seen.
 *  Int 'cur_dim' tracks nested array rows.
 */
class array_dims {
 public:
  std::vector<size_t> dims;
  std::vector<size_t> dims_acc;
  int cur_dim;
  array_dims() : dims(), dims_acc(), cur_dim(0) {}

  bool operator==(const array_dims& other) {
    return dims == other.dims && dims_acc == other.dims_acc
           && cur_dim == other.cur_dim;
  }

  bool operator!=(const array_dims& other) { return !operator==(other); }

  std::string print() {
    // only used for debugging
    std::stringstream ss;
    ss << " num dims: " << dims.size() << "\tdim szs: ";
    for (auto& x : dims)
      ss << " " << x;
    ss << "\tdims_acc cts: ";
    for (auto& x : dims_acc)
      ss << " " << x;
    ss << std::endl;
    return ss.str();
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
  vars_map_r& vars_r;
  vars_map_i& vars_i;
  std::vector<std::string> key_stack;
  std::map<std::string, int> var_types_map;   // vars_r and vars_i entries
  std::map<std::string, int> slot_types_map;  // all slots all vars parsed
  std::map<std::string, array_dims> slot_dims_map;
  std::map<std::string, bool> int_slots_map;
  std::vector<double> values_r;
  std::vector<int> values_i;
  size_t array_start_i;  // index into values_i
  size_t array_start_r;  // index into values_r
  int event;

  void reset_values() {
    values_r.clear();
    values_i.clear();
    array_start_i = 0;
    array_start_r = 0;
  }

  bool is_init() {
    return (key_stack.empty() && var_types_map.empty() && slot_types_map.empty()
            && values_r.empty() && values_i.empty() && slot_dims_map.empty()
            && array_start_i == 0 && array_start_r == 0
            && int_slots_map.empty());
  }

  std::string key_str() { return boost::algorithm::join(key_stack, "."); }

  std::string outer_key_str() {
    std::string result;
    if (key_stack.size() > 1) {
      std::string slot = key_stack.back();
      key_stack.pop_back();
      result = key_str();
      key_stack.push_back(slot);
    }
    return result;
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
  json_data_handler(vars_map_r& a_vars_r, vars_map_i& a_vars_i)
      : json_handler(),
        vars_r(a_vars_r),
        vars_i(a_vars_i),
        key_stack(),
        var_types_map(),
        slot_types_map(),
        slot_dims_map(),
        int_slots_map(),
        values_r(),
        values_i(),
        array_start_i(0),
        array_start_r(0) {}

  // *** start handler events ***
  void start_text() {
    vars_i.clear();  // can't accumulate var defs across calls to parser
    vars_r.clear();
    var_types_map.clear();
    slot_types_map.clear();
    reset_values();
  }

  void end_text() {
    save_key_value_pair();
    convert_arrays();
    reset_values();
  }

  void key(const std::string& key) {
    if (event != meta_event::OBJ_OPEN) {
      save_key_value_pair();
    }
    event = meta_event::KEY;
    reset_values();
    key_stack.push_back(key);
    if (slot_types_map.count(key_str()) == 0) {
      slot_types_map[key_str()] = meta_type::SCALAR;
      int_slots_map[key_str()] = true;
    }
  }

  void start_object() {
    event = meta_event::OBJ_OPEN;
    if (is_init())
      return;
    if (slot_types_map[key_str()] == meta_type::ARRAY) {
      slot_types_map[key_str()] = meta_type::ARRAY_OF_TUPLES;
    } else if (slot_types_map[key_str()] == meta_type::SCALAR) {
      slot_types_map[key_str()] = meta_type::TUPLE;
    }
    // don't reset tuples or array of tuples
  }

  void end_object() {
    event = meta_event::OBJ_CLOSE;
    if (key_stack.size() > 1
        && slot_types_map[outer_key_str()] == meta_type::ARRAY_OF_TUPLES) {
      array_dims outer = get_outer_dims(key_stack);
      if (!outer.dims.empty()) {
        outer.dims_acc[outer.dims.size() - 1]++;
        set_outer_dims(outer);
      }
    }
    save_key_value_pair();
  }

  void start_array() {
    if (key_stack.empty()) {
      throw json_error("expecting JSON object, found array");
    }
    if (slot_types_map[key_str()] == meta_type::SCALAR
        && !(values_r.empty() && values_r.empty())) {
      std::stringstream errorMsg;
      errorMsg << "variable: " << key_str()
               << ", error: non-scalar array value";
      throw json_error(errorMsg.str());
    }
    if (slot_types_map[key_str()] == meta_type::SCALAR)
      slot_types_map[key_str()] = meta_type::ARRAY;
    else if (slot_types_map[key_str()] == meta_type::TUPLE)
      unexpected_error(key_str());
    array_dims dims;
    if (slot_dims_map.count(key_str()) == 1)
      dims = slot_dims_map[key_str()];
    dims.cur_dim++;
    if (dims.dims.empty() || dims.dims.size() < dims.cur_dim) {
      dims.dims.push_back(0);
      dims.dims_acc.push_back(0);
    }
    if (dims.cur_dim > 1)
      dims.dims_acc[dims.cur_dim - 2]++;
    slot_dims_map[key_str()] = dims;
    array_start_i = values_i.size();
    array_start_r = values_r.size();
  }

  void end_array() {
    if (slot_dims_map.count(key_str()) == 0)
      unexpected_error(key_str());
    array_dims dims = slot_dims_map[key_str()];
    int idx = dims.cur_dim - 1;
    bool is_int = int_slots_map[key_str()];
    bool is_last = (slot_types_map[key_str()] != meta_type::ARRAY_OF_TUPLES
                    && dims.cur_dim == dims.dims.size());
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
        errorMsg << "variable: " << key_str()
                 << ", error: non-rectangular array";
        throw json_error(errorMsg.str());
      }
    }
    dims.dims_acc[idx] = 0;
    dims.cur_dim--;
    slot_dims_map[key_str()] = dims;
  }

  void null() {
    std::stringstream errorMsg;
    errorMsg << "variable: " << key_str() << ", error: null values not allowed";
    throw json_error(errorMsg.str());
  }

  void boolean(bool p) {
    std::stringstream errorMsg;
    errorMsg << "variable: " << key_str()
             << ", error: boolean values not allowed";
    throw json_error(errorMsg.str());
  }

  void string(const std::string& s) {
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
      errorMsg << "variable: " << key_str()
               << ", error: string values not allowed";
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
    if (int_slots_map[key_str()]) {
      values_i.push_back(n);
    } else {
      values_r.push_back(n);
    }
  }

  void number_unsigned_int(unsigned n) {
    // if integer overflow, promote numeric data to double
    if (n > (unsigned)std::numeric_limits<int>::max())
      promote_to_double();
    if (int_slots_map[key_str()]) {
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
    if (int_slots_map[key_str()]) {
      for (std::vector<int>::iterator it = values_i.begin();
           it != values_i.end(); ++it)
        values_r.push_back(*it);
      array_start_r = array_start_i;
    }
    int_slots_map[key_str()] = false;
  }

  /* Save non-tuple vars and innermost tuple slots to vars_i and vars_r.
   * For arrays of tuples we need to check that new elements are consistent
   * with previous tuple elements.
   */
  void save_key_value_pair() {
    if (0 == key_stack.size())
      return;
    if (slot_types_map.count(key_str()) < 1)
      unexpected_error(key_str());
    if (slot_types_map[key_str()] == meta_type::TUPLE
        || slot_types_map[key_str()] == meta_type::ARRAY_OF_TUPLES) {
      {}
    } else {
      bool is_int = int_slots_map[key_str()];
      bool is_new
          = (vars_r.count(key_str()) == 0 && vars_i.count(key_str()) == 0);
      bool is_real = vars_r.count(key_str()) == 1;
      bool was_int = vars_i.count(key_str()) == 1;
      std::vector<size_t> dims;
      if (slot_dims_map.count(key_str()) == 1)
        dims = slot_dims_map[key_str()].dims;
      if (is_new) {
        var_types_map[key_str()] = slot_types_map[key_str()];
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
        if (!is_array_tuples(key_stack)) {
          std::stringstream errorMsg;
          errorMsg << "attempt to redefine variable: " << key_str();
          throw json_error(errorMsg.str());
        }
        var_types_map[key_str()] = meta_type::ARRAY;
        std::vector<size_t> dims = slot_dims_map[key_str()].dims;
        if ((!is_int && was_int) || (is_int && is_real)) {  // promote to double
          std::vector<double> values_tmp;
          for (auto& x : vars_i[key_str()].first) {
            values_tmp.push_back(x);
          }
          for (auto& x : values_r)
            values_tmp.push_back(x);
          std::pair<std::vector<double>, std::vector<size_t>> pair;
          pair = make_pair(values_tmp, dims);
          vars_r[key_str()] = pair;
          vars_i.erase(key_str());
        } else if (is_int) {
          for (auto& x : values_i)
            vars_i[key_str()].first.push_back(x);
          vars_i[key_str()].second = dims;
        } else {
          for (auto& x : values_r)
            vars_r[key_str()].first.push_back(x);
          vars_r[key_str()].second = dims;
        }
      }
    }
    key_stack.pop_back();
  }

  /* Process array variables
   *  a. for array of tuples, concatenate dimensions
   *  b. convert vector of values in row-major order
   *     to vector of values in column-major order.
   * Update vars_i and vars_r accordingly.
   */
  void convert_arrays() {
    for (auto const& var : var_types_map) {
      if (var.second != meta_type::ARRAY) {
        continue;
      }
      std::vector<size_t> all_dims;
      array_dims inner = slot_dims_map[var.first];
      // MM TODO: do we need to generalize to more deeply nested structures?
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
          to_column_major(var.first, cm_values_i,
                          vars_i[var.first].first, vars_i[var.first].second);
          pair = make_pair(cm_values_i, vars_i[var.first].second);
        } else {
          to_column_major(var.first, cm_values_i,
                          vars_i[var.first].first, all_dims);
          pair = make_pair(cm_values_i, all_dims);
        }
        vars_i[var.first] = pair;
      } else if (vars_r.count(var.first) == 1) {
        std::vector<double> cm_values_r(vars_r[var.first].first.size());
        std::pair<std::vector<double>, std::vector<size_t>> pair;
        if (all_dims.empty()) {
          to_column_major(var.first, cm_values_r,
                          vars_r[var.first].first, vars_r[var.first].second);
          pair = make_pair(cm_values_r, vars_r[var.first].second);
        } else {
          to_column_major(var.first, cm_values_r,
                          vars_r[var.first].first, all_dims);
          pair = make_pair(cm_values_r, all_dims);
        }
        vars_r[var.first] = pair;
      } else {
          std::stringstream errorMsg;
          errorMsg << "variable: " << var.first << ", ill-formed json";
          throw json_error(errorMsg.str());
      }
    }
  }

  template <typename T>
  void to_column_major(std::string vname,
                       std::vector<T>& cm_vals,
                       const std::vector<T>& rm_vals,
                       const std::vector<size_t>& dims) {
    size_t expected_size = 1;
    for (auto&x : dims)
      expected_size *= x;
    if (expected_size != rm_vals.size()) {
      std::stringstream errorMsg;
      errorMsg << "variable: " << vname << ", error: ill-formed array";
      throw json_error(errorMsg.str());
    }

    for (size_t i = 0; i < rm_vals.size(); i++) {
      size_t idx = convert_offset_rtl_2_ltr(vname, i, dims);
      cm_vals[idx] = rm_vals[i];
    }
  }

  // convert row-major offset to column-major offset
  size_t convert_offset_rtl_2_ltr(std::string vname,
                                  size_t rtl_offset,
                                  const std::vector<size_t>& dims) {
    size_t rtl_dsize = 1;
    for (size_t i = 1; i < dims.size(); i++)
      rtl_dsize *= dims[i];

    // double-check array indexing
    if (rtl_offset >= rtl_dsize * dims[0]) {
      std::stringstream errorMsg;
      errorMsg << "variable: " << vname << ", unexpected error";
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
    errorMsg << "unexpected parsing error, at key " << where;
    throw json_error(errorMsg.str());
  }

  array_dims get_outer_dims(const std::vector<std::string>& keys) {
    std::vector<std::string> stack = keys;
    std::string key;
    stack.pop_back();
    while (!stack.empty()) {
      key = boost::algorithm::join(stack, ".");
      if (slot_dims_map.count(key) == 1)
        return slot_dims_map[key];
      stack.pop_back();
    }
    key = boost::algorithm::join(keys, ".");
    if (slot_dims_map.count(key) != 1)
      unexpected_error(key);
    return slot_dims_map[key];
  }

  bool is_array_tuples(const std::vector<std::string>& keys) {
    std::vector<std::string> stack = keys;
    std::string key;
    stack.pop_back();
    while (!stack.empty()) {
      key = boost::algorithm::join(stack, ".");
      if (slot_types_map[key] == meta_type::ARRAY_OF_TUPLES)
        return true;
      stack.pop_back();
    }
    return false;
  }

  void set_outer_dims(array_dims update) {
    std::vector<std::string> stack = key_stack;
    std::string key;
    stack.pop_back();
    while (!stack.empty()) {
      key = boost::algorithm::join(stack, ".");
      if (slot_dims_map.count(key) == 1)
        break;
      stack.pop_back();
    }
    if (stack.empty()) {
      key = boost::algorithm::join(key_stack, ".");
      unexpected_error(key);
    }
    slot_dims_map[key] = update;
  }

  // for debugging only
  void dump_state(std::string where) {
    std::string slot_type("unknown");
    if (slot_types_map.count(key_str()) == 1)
      slot_type = std::to_string(slot_types_map[key_str()]);
    bool is_int = true;
    if (int_slots_map.count(key_str()) == 1)
      is_int = int_slots_map[key_str()];
    std::cout << where << " key " << key_str() << " slot_type " << slot_type
              << " is_int " << is_int << "\n\tvalues_i (" << values_i.size()
              << ") ";
    for (auto& x : values_i)
      std::cout << " " << x;
    std::cout << "\n\tvalues_r (" << values_r.size() << ") ";
    for (auto& x : values_r)
      std::cout << " " << x;
    std::cout << std::endl;
    if (slot_dims_map.count(key_str()) == 1)
      std::cout << slot_dims_map[key_str()].print();
    else
      std::cout << std::endl;
    std::cout << "\tknown int vars (" << vars_i.size() << ") ";
    for (auto& x : vars_i)
      std::cout << " " << x.first;
    std::cout << "\tknown real vars (" << vars_r.size() << ") ";
    for (auto& x : vars_r)
      std::cout << " " << x.first;
    std::cout << std::endl;
  }
};

}  // namespace json

}  // namespace stan

#endif
