#ifndef STAN_IO_JSON_RAPIDJSON_PARSER_HPP
#define STAN_IO_JSON_RAPIDJSON_PARSER_HPP


#include "rapidjson/reader.h"
#include "rapidjson/document.h"
#include "rapidjson/istreamwrapper.h"
#include "rapidjson/cursorstreamwrapper.h"
#include "rapidjson/error/en.h"

#include <stan/io/validate_zero_buf.hpp>
#include <stan/io/json/json_error.hpp>
#include <stdexcept>
#include <iostream>
#include <istream>
#include <limits>
#include <sstream>
#include <string>
#include <iostream>
#include <chrono>
#include <fstream>
#include <string>
#include <cerrno>


namespace stan {
    namespace json {
        enum class ParsingState { Idle, Started };

        template<typename Handler>
        struct RapidJSONHandler {
            RapidJSONHandler(Handler& h) : h_(h), state_(ParsingState::Idle) {
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
            bool Null() {
                h_.null();
                return true; 
            }
            bool Bool(bool b) {
                h_.boolean(b);
                return true;
            }
            bool Int(int i) {
                h_.number_long(i);
                return true;
            }
            bool Uint(unsigned u) {
                h_.number_unsigned_long(u);
                return true;
            }
            bool Int64(int64_t i) {
                h_.number_long(i);
                return true;
            }
            bool Uint64(uint64_t u) {
                h_.number_unsigned_long(u);
                return true;
            }
            bool Double(double d) {
                h_.number_double(d);
                return true;
            }
            bool RawNumber(const char* str, rapidjson::SizeType length, bool copy) { 
                std::cout << "ERROR" << std::endl;
                // this probably should not happen in our case
                return true;
            }
            bool String(const char* str, rapidjson::SizeType length, bool copy) {
                h_.string(str);
                return true;
            }
            bool StartObject() { 
                if(state_ == ParsingState::Idle) {
                    state_ = ParsingState::Started;
                    h_.start_object();
                    return true;
                }else{
                    // error check here, nested objects not allowed
                    return true;
                }                
            }
            bool Key(const char* str, rapidjson::SizeType length, bool copy) {
                h_.key(str);
                return true;
            }
            bool EndObject(rapidjson::SizeType memberCount) {
                // This should be the end as nested objects are not allowed
                h_.end_object();
                return true;
            }
            bool StartArray() { 
                h_.start_array();                
                return true; 
            }
            bool EndArray(rapidjson::SizeType elementCount) { 
                h_.end_array();
                return true;
            }

            Handler& h_;        
            ParsingState state_;
            std::string error_message;
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
                   
            if (!reader.Parse(isw, filter)) {
                rapidjson::ParseErrorCode err = reader.GetParseErrorCode();
                std::stringstream ss;
                ss << "Error in JSON parsing at"
                << std::endl << rapidjson::GetParseError_En(err) << std::endl;
                throw json_error(ss.str());
            }
            /*else{
                if(reader.HasParseError()){
                    std::cout << "OK" << rapidjson::GetParseError_En(reader.GetParseErrorCode()) << std::endl;
                }
                std::cout << "OKI" << std::endl;
                
            }*/
        }
    }
}
#endif
