#ifndef STAN_IO_JSON_RAPIDJSON_PARSER_HPP
#define STAN_IO_JSON_RAPIDJSON_PARSER_HPP

#include <rapidjson/reader.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/error/en.h>
#include <rapidjson/encodings.h>
#include <stan/io/validate_zero_buf.hpp>
#include <stan/io/json/json_error.hpp>

#include <cerrno>
#include <stdexcept>
#include <iostream>
#include <istream>
#include <limits>
#include <sstream>
#include <string>
#include <fstream>





namespace stan {
    namespace json {
        enum class ParsingState { Idle, Started, End };

        template<typename Handler>
        struct RapidJSONHandler {
            explicit RapidJSONHandler(Handler& h) : h_(h),
                state_(ParsingState::Idle) {
            }
            json_error json_exception(const std::string& msg) const {
                std::stringstream ss;
                ss << "Error in JSON parsing at"
                // << " line=" << line_ << " column=" << column_
                << std::endl
                << msg
                << std::endl;
                return json_error(ss.str());
            }
            void check_start() {
                if (state_ == ParsingState::Idle) {
                    json_exception(
                        "expecting start of object ({) or array ([)\n");
                }
            }
            bool Null() {
                check_start();
                h_.null();
                return true;
            }
            bool Bool(bool b) {
                check_start();
                h_.boolean(b);
                return true;
            }
            bool Int(int i) {
                check_start();
                h_.number_long(i);
                return true;
            }
            bool Uint(unsigned u) {
                check_start();
                h_.number_unsigned_long(u);
                return true;
            }
            bool Int64(int64_t i) {
                check_start();
                h_.number_long(i);
                return true;
            }
            bool Uint64(uint64_t u) {
                check_start();
                h_.number_unsigned_long(u);
                return true;
            }
            bool Double(double d) {
                check_start();
                h_.number_double(d);
                return true;
            }
            bool RawNumber(const char* str, rapidjson::SizeType length,
                    bool copy) {
                // this will never get
                return true;
            }
            bool String(const char* str, rapidjson::SizeType length,
                    bool copy) {
                check_start();
                h_.string(str);
                return true;
            }
            bool StartObject() {
                check_start();
                state_ = ParsingState::Started;
                h_.start_object();
                return true;
            }
            bool Key(const char* str, rapidjson::SizeType length, bool copy) {
                check_start();
                h_.key(str);
                last_key_ = str;
                return true;
            }
            bool EndObject(rapidjson::SizeType memberCount) {
                check_start();
                h_.end_object();
                return true;
            }
            bool StartArray() {
                check_start();
                h_.start_array();
                return true;
            }
            bool EndArray(rapidjson::SizeType elementCount) {
                check_start();
                h_.end_array();
                return true;
            }

            Handler& h_;
            ParsingState state_;
            std::string error_message;
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
        void rapidjson_parse(std::istream& in,
                Handler& handler) {
            rapidjson::Reader reader;
            RapidJSONHandler<Handler> filter(handler);
            rapidjson::IStreamWrapper isw(in);
            handler.start_text();
            if (!reader.Parse<rapidjson::kParseNanAndInfFlag |
                rapidjson::kParseValidateEncodingFlag |
                rapidjson::kParseFullPrecisionFlag>(isw, filter)) {
                rapidjson::ParseErrorCode err = reader.GetParseErrorCode();
                std::stringstream ss;
                ss << "Error in JSON parsing " << std::endl
                << rapidjson::GetParseError_En(err) << std::endl;
                throw json_error(ss.str());
            }
            handler.end_text();
        }
    }
}
#endif
