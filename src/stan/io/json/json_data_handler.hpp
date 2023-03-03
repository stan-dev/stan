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
#include <boost/regex.hpp>

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

/** Enum which tracks handler events.
    Used to identify first and last slot of a tuple.
*/
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
};

/** Tracks num slots in a tuple for array of tuple consistency.
 */
class tuple_slots {
 public:
  size_t slots;
  size_t slots_acc;
  bool is_first;
  tuple_slots() : slots(0), slots_acc(0), is_first(true) {}
};

/**
 * A <code>json_data_handler</code> is an implementation of a
 * <code>json_handler</code> that restricts the allowed JSON text
 * to a single JSON object which define Stan variables.
 * The handler is responsible for populating the data structures
 * `vars_r` and `vars_i` which map Stan variable names to the values
 * and dimensions found in the JSON.
 *
 * In the JSON, each Stan variable is a JSON key : value pair.
 * The key is a string (the Stan variable name) and the value
 * is either a scalar variables, array, or a tuple.
 * Stan program variables can only be of type int or real (double).
 * The strings \"Inf\" and \"Infinity\" are mapped to positive infinity,
 * the strings \"-Inf\" and \"-Infinity\" are mapped to negative infinity,
 * and the string \"NaN\" is mapped to not-a-number.
 * Bare versions of Infinity, -Infinity, and NaN are also allowed.
 *
 * Tuple variables consist of a JSON object, whose keys correspond
 * to the slot number, counting from 1.  Tuple elements can be arrays
 * or other tuples and array elements can be tuples, which allows
 * for any level of nested arrays within tuples, or tuples within arrays.
 *
 * For a Stan model variable which has nested tuples, only the innermost
 * tuple slot will correspond to a variable in the generated C++ code,
 * likewise, in the JSON object, only the innermost elements will be
 * int or real values. For arrays of tuples, the handler needs to track
 * both the array dimension and whether or not the values found so far
 * are of type real or int.  To do this, the handler uses a series of
 * maps between tuple slots seen so far and the C++ storage type (int or real),
 * the variable meta-type (array, tuple, or array of tuples).
 * For arrays of tuples, we need to ensure that all tuple elements of an array
 * are of the same shape.  To do this we track the number of slots in the tuple
 * as well as the dimensions of any array slots in the tuple.
 *
 * If the top-level object entry key is not a legal Stan variable name
 * the handler will not check the corresponding value, other than maintining
 * the event state and key_stack.
 */
class json_data_handler : public stan::json::json_handler {
 private:
  vars_map_r& vars_r;
  vars_map_i& vars_i;
  std::vector<std::string> key_stack;
  std::map<std::string, int> var_types_map;   // vars_r and vars_i entries
  std::map<std::string, int> slot_types_map;  // all slots all vars parsed
  std::map<std::string, array_dims> slot_dims_map;
  std::map<std::string, tuple_slots> tuple_slots_map;
  std::map<std::string, bool> int_slots_map;
  std::vector<double> values_r;  // accumulates real var values
  std::vector<int> values_i;     // accumulates int var values
  size_t array_start_i;          // index into values_i
  size_t array_start_r;          // index into values_r
  int event;                     // tracks most recent meta_event
  bool not_stan_var;             // accept non-Stan entries

  void reset_values() {
    // Once var values have been copied into var_context maps,
    // clear the accumulator vectors.
    values_r.clear();
    values_i.clear();
    array_start_i = 0;
    array_start_r = 0;
  }

  inline std::string key_str() {
    return boost::algorithm::join(key_stack, ".");
  }

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

  bool is_init() {
    return (key_stack.empty() && var_types_map.empty() && slot_types_map.empty()
            && values_r.empty() && values_i.empty() && slot_dims_map.empty()
            && array_start_i == 0 && array_start_r == 0
            && int_slots_map.empty());
  }

  /** Stan variable names must start with a letter
   *  and contain only letters, numbers, or an underscore.
   */
  bool valid_varname(const std::string& name) {
    static const boost::regex re("[a-zA-Z][a-zA-Z0-9_]*");
    return regex_match(name, re);
  }

  bool is_array_tuples(const std::vector<std::string>& keys) {
    std::vector<std::string> stack(keys);
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

  array_dims get_outer_dims(const std::vector<std::string>& keys) {
    std::vector<std::string> stack(keys);
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
      unexpected_error(key, "not an array");
    return slot_dims_map[key];
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
      unexpected_error(key, "ill-formed array");
    }
    slot_dims_map[key] = update;
  }

  void promote_to_double() {
    if (int_slots_map[key_str()]) {
      int_slots_map[key_str()] = false;
      values_r.reserve(values_i.size());
      values_r.insert(values_r.end(), values_i.begin(), values_i.end());
      array_start_r = array_start_i;
      values_i.clear();
      array_start_i = 0;
    }
  }

  /* Save non-tuple vars and innermost tuple slots to vars_i and vars_r.
   * Converts multi-dim arrays from row-major to column major.
   * For arrays of tuples we need to check that new elements are consistent
   * with previous tuple elements.
   */
  void save_key_value_pair() {
    if (key_stack.empty())
      return;
    if (not_stan_var) {
      key_stack.pop_back();
      return;
    }
    std::string key = key_str();
    if (slot_types_map.count(key) < 1)
      unexpected_error(key, "unknown variable");
    if (slot_types_map[key] == meta_type::SCALAR
        || slot_types_map[key] == meta_type::ARRAY) {
      bool is_new = (vars_r.count(key) == 0 && vars_i.count(key) == 0);
      bool is_int = int_slots_map[key];
      bool is_real = vars_r.count(key) == 1;
      bool was_int = !is_int && vars_i.count(key) == 1;
      std::vector<size_t> dims;
      if (slot_dims_map.count(key) == 1)
        dims = slot_dims_map[key].dims;
      if (dims.size() > 1) {
        if (is_int) {
          std::vector<int> cm_values_i(values_i.size());
          to_column_major(key, cm_values_i, values_i, dims);
          values_i.assign(cm_values_i.begin(), cm_values_i.end());
        } else {
          std::vector<double> cm_values_r(values_r.size());
          to_column_major(key, cm_values_r, values_r, dims);
          values_r.assign(cm_values_r.begin(), cm_values_r.end());
        }
      }
      if (is_new) {
        var_types_map[key] = slot_types_map[key];
        if (is_int) {
          std::pair<std::vector<int>, std::vector<size_t>> pair;
          pair = make_pair(values_i, dims);
          vars_i[key] = pair;
        } else {
          std::pair<std::vector<double>, std::vector<size_t>> pair;
          pair = make_pair(values_r, dims);
          vars_r[key] = pair;
        }
      } else {
        bool is_aot = false;
        std::string vname;
        for (auto& key : key_stack) {
          vname.append(key);
          if (slot_types_map[vname] == meta_type::ARRAY_OF_TUPLES) {
            is_aot = true;
            break;
          }
          vname.append(".");
        }
        if (!is_aot)
          unexpected_error(key, "not array of tuples");
        bool consistent = true;
        if (is_int || was_int) {
          auto expect_dims = vars_i[key].second;
          size_t expect_vals_len = 1;
          for (auto& x : expect_dims)
            expect_vals_len *= x;
          if (is_int) {
            if (expect_vals_len != values_i.size())
              consistent = false;
          } else {
            if (expect_vals_len != values_r.size())
              consistent = false;
          }
        } else {
          auto expect_dims = vars_r[key].second;
          size_t expect_vals_len = 1;
          for (auto& x : expect_dims)
            expect_vals_len *= x;
          if (expect_vals_len != values_r.size())
            consistent = false;
        }
        if (!consistent) {
          std::stringstream errorMsg;
          errorMsg << "Variable " << key
                   << ": size mismatch between tuple elements.";
          throw json_error(errorMsg.str());
        }
        var_types_map[key] = meta_type::ARRAY;
        if ((!is_int && was_int) || (is_int && is_real)) {  // promote to double
          std::vector<double> values_tmp;
          for (auto& x : vars_i[key].first) {
            values_tmp.push_back(x);
          }
          for (auto& x : values_r)
            values_tmp.push_back(x);
          std::pair<std::vector<double>, std::vector<size_t>> pair;
          pair = make_pair(values_tmp, dims);
          vars_r[key] = pair;
          vars_i.erase(key);
        } else if (is_int) {
          for (auto& x : values_i)
            vars_i[key].first.push_back(x);
          vars_i[key].second = dims;
        } else {
          for (auto& x : values_r)
            vars_r[key].first.push_back(x);
          vars_r[key].second = dims;
        }
      }
    }
    key_stack.pop_back();
  }

  /* For array of tuples, concatenate dimensions
   * Update vars_i and vars_r dimensions accordingly.
   */
  void update_array_dims() {
    for (auto const& var : var_types_map) {
      if (var.second != meta_type::ARRAY) {
        continue;
      }
      std::vector<size_t> all_dims;
      std::vector<std::string> slots;
      split(slots, var.first, boost::is_any_of("."), boost::token_compress_on);
      std::string slot;
      for (size_t i = 0; i < slots.size(); ++i) {
        slot.append(slots[i]);
        if (slot_dims_map.count(slot) == 1
            && !slot_dims_map[slot].dims.empty()) {
          for (auto& x : slot_dims_map[slot].dims)
            all_dims.push_back(x);
        }
        slot.append(".");
      }
      if (vars_i.count(var.first) == 1) {
        if (all_dims.size() == vars_i[var.first].second.size())
          continue;
        else
          vars_i[var.first].second.assign(all_dims.begin(), all_dims.end());
      } else if (vars_r.count(var.first) == 1) {
        if (all_dims.size() == vars_r[var.first].second.size())
          continue;
        else
          vars_r[var.first].second.assign(all_dims.begin(), all_dims.end());
      } else {
        std::stringstream errorMsg;
        errorMsg << "Variable: " << var.first << ", ill-formed JSON.";
        throw json_error(errorMsg.str());
      }
    }
  }

  template <typename T>
  void to_column_major(std::string vname, std::vector<T>& cm_vals,
                       const std::vector<T>& rm_vals,
                       const std::vector<size_t>& dims) {
    size_t expected_size = 1;
    for (auto& x : dims)
      expected_size *= x;
    if (expected_size != rm_vals.size()) {
      std::stringstream errorMsg;
      errorMsg << "Variable: " << vname << ", error: ill-formed array.";
      throw json_error(errorMsg.str());
    }
    for (size_t i = 0; i < rm_vals.size(); i++) {
      size_t idx = convert_offset_rtl_2_ltr(vname, i, dims);
      cm_vals[idx] = rm_vals[i];
    }
  }

  void unexpected_error(const std::string& where, const std::string& what) {
    std::stringstream errorMsg;
    errorMsg << "Variable " << where << ", " << what << ".";
    throw json_error(errorMsg.str());
  }

 public:
  /**
   * Construct a json_data_handler object.
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
        tuple_slots_map(),
        int_slots_map(),
        values_r(),
        values_i(),
        array_start_i(0),
        array_start_r(0) {}

  /** Clear all maps before parsing next JSON object.
   *  This means that we don't accumulate variable definitions
   *  across calls to the parser.
   */
  void start_text() {
    vars_i.clear();
    vars_r.clear();
    var_types_map.clear();
    slot_types_map.clear();
    slot_dims_map.clear();
    tuple_slots_map.clear();
    int_slots_map.clear();
    reset_values();
    not_stan_var = true;
  }

  /** Once all variable definitions have been processed,
   *  update dimensions for array of tuple variables.
   */
  void end_text() { update_array_dims(); }

  /** A key is either a top-level Stan variable name or a tuple slot id.
   *  Logic handles edge case where key is the first slot of a tuple;
   *  the name of the enclosing object is not used in the generated C++,
   *  but we still need to track the number of tuple slots.
   */
  void key(const std::string& key) {
    if (event != meta_event::OBJ_OPEN) {
      save_key_value_pair();
    }
    event = meta_event::KEY;
    reset_values();
    std::string outer = key_str();
    key_stack.push_back(key);
    if (key_stack.size() == 1) {
      not_stan_var = !valid_varname(key);
    }
    if (not_stan_var)
      return;
    if (key_stack.size() == 1 && slot_types_map.count(key) == 1) {
      std::stringstream errorMsg;
      errorMsg << "Attempt to redefine variable: " << key << ".";
      throw json_error(errorMsg.str());
    } else if (key_stack.size() > 1
               && slot_types_map[outer] == meta_type::ARRAY_OF_TUPLES) {
      if (tuple_slots_map[outer].is_first) {
        tuple_slots_map[outer].slots++;
      } else {
        tuple_slots_map[outer].slots_acc++;
      }
    }
    std::string vname = key_str();
    if (slot_types_map.count(vname) == 0) {
      slot_types_map[vname] = meta_type::SCALAR;
      int_slots_map[vname] = true;
    }
  }

  /**
   * A start object ("{") event changes the meta-type of the current key.
   * Initialize or update tuple slots.
   */
  void start_object() {
    event = meta_event::OBJ_OPEN;
    if (is_init() || not_stan_var)
      return;
    std::string key = key_str();
    if (slot_types_map[key] == meta_type::ARRAY) {
      slot_types_map[key] = meta_type::ARRAY_OF_TUPLES;
    } else if (slot_types_map[key_str()] == meta_type::SCALAR) {
      slot_types_map[key_str()] = meta_type::TUPLE;
    }
    if (slot_types_map[key] == meta_type::ARRAY_OF_TUPLES) {
      if (tuple_slots_map.count(key) == 0) {
        tuple_slots slots;
        tuple_slots_map[key] = slots;
      } else {
        tuple_slots_map[key].is_first = false;
        tuple_slots_map[key].slots_acc = 0;
      }
    }
  }

  /** An end object ("}") event closes either the top-level object or a tuple.
   *  If this is an array of tuples, track or check the number tuple slots
   *  and the array size.
   */
  void end_object() {
    event = meta_event::OBJ_CLOSE;
    if (not_stan_var) {
      if (!key_stack.empty())
        key_stack.pop_back();
      return;
    }
    if (key_stack.size() > 1) {
      std::string tuple = outer_key_str();
      if (slot_types_map[tuple] == meta_type::ARRAY_OF_TUPLES) {
        array_dims outer = get_outer_dims(key_stack);
        if (!outer.dims.empty()) {
          outer.dims_acc[outer.dims.size() - 1]++;
          set_outer_dims(outer);
        }
        if (tuple_slots_map.count(tuple) == 0)
          unexpected_error(tuple, "found close object, not a tuple var");
        if (tuple_slots_map[tuple].is_first) {
          tuple_slots_map[tuple].is_first = false;
        } else {
          if (tuple_slots_map[tuple].slots_acc
              != tuple_slots_map[tuple].slots) {
            std::stringstream errorMsg;
            errorMsg << "Variable " << tuple
                     << ": size mismatch between tuple elements.";
            throw json_error(errorMsg.str());
          }
        }
      }
    }
    save_key_value_pair();
  }

  /** For a start array ("[") event we first check that we're not currently
   *  processing the values in an array.  We need to do this because JSON
   *  doesn't distinguish lists of heterogenous elements and arrays.
   *  Then we add or update the dimensions of the array variable.
   */
  void start_array() {
    if (key_stack.empty()) {
      throw json_error("Expecting JSON object, found array.");
    }
    if (not_stan_var)
      return;
    std::string key(key_str());
    if (slot_types_map[key] == meta_type::SCALAR
        && !(values_r.empty() && values_r.empty())) {
      std::stringstream errorMsg;
      errorMsg << "Variable: " << key << ", error: non-scalar array value.";
      throw json_error(errorMsg.str());
    }
    if (slot_types_map[key] == meta_type::SCALAR)
      slot_types_map[key] = meta_type::ARRAY;
    else if (slot_types_map[key] == meta_type::TUPLE)
      unexpected_error(key, "ill-formed tuple");
    array_dims dims;
    if (slot_dims_map.count(key) == 1)
      dims = slot_dims_map[key];
    dims.cur_dim++;
    if (dims.dims.empty() || dims.dims.size() < dims.cur_dim) {
      dims.dims.push_back(0);
      dims.dims_acc.push_back(0);
    }
    if (dims.cur_dim > 1)
      dims.dims_acc[dims.cur_dim - 2]++;
    slot_dims_map[key] = dims;
    array_start_i = values_i.size();
    array_start_r = values_r.size();
  }

  /** An array event ("]") closes the current array dimension.
   *  The innermost array dimension are the scalar elements which
   *  are found in the accumulator vectors `values_i` and `values_r`.
   *  If processing the first row of an array, record the size of this row,
   *  else check that the size of this row matches recorded row size.
   */
  void end_array() {
    if (not_stan_var)
      return;
    if (slot_dims_map.count(key_str()) == 0)
      unexpected_error(key_str(), "ill-formed array");
    std::string key(key_str());
    array_dims dims = slot_dims_map[key];
    int idx = dims.cur_dim - 1;
    bool is_int = int_slots_map[key];
    bool is_last = (slot_types_map[key] != meta_type::ARRAY_OF_TUPLES
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
        errorMsg << "Variable: " << key << ", error: non-rectangular array.";
        throw json_error(errorMsg.str());
      }
    }
    dims.dims_acc[idx] = 0;
    dims.cur_dim--;
    slot_dims_map[key] = dims;
  }

  void null() {
    if (not_stan_var)
      return;
    std::stringstream errorMsg;
    errorMsg << "Variable: " << key_str()
             << ", error: null values not allowed.";
    throw json_error(errorMsg.str());
  }

  void boolean(bool p) {
    if (not_stan_var)
      return;
    std::stringstream errorMsg;
    errorMsg << "Variable: " << key_str()
             << ", error: boolean values not allowed.";
    throw json_error(errorMsg.str());
  }

  void string(const std::string& s) {
    if (not_stan_var)
      return;
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
      errorMsg << "Variable: " << key_str()
               << ", error: string values not allowed.";
      throw json_error(errorMsg.str());
    }
    promote_to_double();
    values_r.push_back(tmp);
  }

  void number_double(double x) {
    if (not_stan_var)
      return;
    promote_to_double();
    values_r.push_back(x);
  }

  void number_int(int n) {
    if (not_stan_var)
      return;
    if (int_slots_map[key_str()]) {
      values_i.push_back(n);
    } else {
      values_r.push_back(n);
    }
  }

  void number_unsigned_int(unsigned n) {
    if (not_stan_var)
      return;
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
    if (not_stan_var)
      return;
    // the number doesn't fit in int (otherwise number_int() would be called)
    number_double(n);
  }

  void number_unsigned_int64(uint64_t n) {
    if (not_stan_var)
      return;
    // the number doesn't fit in int (otherwise number_unsigned_int() would be
    // called)
    number_double(n);
  }

  /** This function provides the column-major offset of an array element
   *  given its row-major offset and the array dimensions.
   */
  size_t convert_offset_rtl_2_ltr(std::string vname, size_t rtl_offset,
                                  const std::vector<size_t>& dims) {
    size_t rtl_dsize = 1;
    for (size_t i = 1; i < dims.size(); i++)
      rtl_dsize *= dims[i];
    if (rtl_offset >= rtl_dsize * dims[0]) {
      std::stringstream errorMsg;
      errorMsg << "Variable: " << vname << ", ill-formed data.";
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
