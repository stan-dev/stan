#ifndef STAN_IO_JSON_RAPIDJSON_PARSER_HPP
#define STAN_IO_JSON_RAPIDJSON_PARSER_HPP

#include <stan/io/json/json_error.hpp>
#include <stan/io/validate_zero_buf.hpp>
#include <rapidjson/encodings.h>
#include <rapidjson/error/en.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/reader.h>

#include <cerrno>
#include <fstream>
#include <iostream>
#include <istream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace stan {
namespace json {
enum class ParsingState { Idle, Started, End };

template <typename Handler>
struct RapidJSONHandler {
  explicit RapidJSONHandler(Handler &h) : h_(h), state_(ParsingState::Idle) {}
  bool check_start() {
    if (state_ == ParsingState::Idle) {
      error_message_ = "expecting start of object ({) or array ([)";
      return false;
    }
    return true;
  }
  bool Null() {
    h_.null();
    return check_start();
  }
  bool Bool(bool b) {
    h_.boolean(b);
    return check_start();
  }
  bool Int(int i) {
    h_.number_int(i);
    return check_start();
  }
  bool Uint(unsigned u) {
    h_.number_unsigned_int(u);
    return check_start();
  }
  bool Int64(int64_t i) {
    h_.number_int64(i);
    return check_start();
  }
  bool Uint64(uint64_t u) {
    h_.number_unsigned_int64(u);
    return check_start();
  }
  bool Double(double d) {
    h_.number_double(d);
    return check_start();
  }
  bool RawNumber(const char *str, rapidjson::SizeType length, bool copy) {
    // this will never get
    return true;
  }
  bool String(const char *str, rapidjson::SizeType length, bool copy) {
    h_.string(str);
    return check_start();
  }
  bool StartObject() {
    state_ = ParsingState::Started;
    error_message_ = "";
    h_.start_object();
    return true;
  }
  bool Key(const char *str, rapidjson::SizeType length, bool copy) {
    h_.key(str);
    last_key_ = str;
    return check_start();
  }
  bool EndObject(rapidjson::SizeType memberCount) {
    h_.end_object();
    return true;
  }
  bool StartArray() {
    state_ = ParsingState::Started;
    error_message_ = "";
    h_.start_array();
    return true;
  }
  bool EndArray(rapidjson::SizeType elementCount) {
    h_.end_array();
    return check_start();
  }

  Handler &h_;
  ParsingState state_;
  std::string error_message_;
  std::string last_key_;
};

/**
 * Parse the JSON text represented by the specified input stream,
 * sending events to the specified handler.
 *
 * @tparam Handler
 * @param in Input stream from which to parse
 * @param handler Handler for events from parser
 */
template <typename Handler>
void rapidjson_parse(std::istream &in, Handler &handler) {
  rapidjson::Reader reader;
  RapidJSONHandler<Handler> filter(handler);
  rapidjson::IStreamWrapper isw(in);
  handler.start_text();
  if (!reader.Parse<rapidjson::kParseNanAndInfFlag
                    | rapidjson::kParseValidateEncodingFlag
                    | rapidjson::kParseFullPrecisionFlag>(isw, filter)) {
    rapidjson::ParseErrorCode err = reader.GetParseErrorCode();
    std::stringstream ss;
    ss << "Error in JSON parsing " << std::endl
       << "at offset " << reader.GetErrorOffset() << ": " << std::endl;
    if (filter.error_message_.size() > 0) {
      ss << filter.error_message_ << std::endl;
    } else {
      ss << rapidjson::GetParseError_En(err) << std::endl;
    }
    throw json_error(ss.str());
  }
  handler.end_text();
}
}  // namespace json
}  // namespace stan
#endif
