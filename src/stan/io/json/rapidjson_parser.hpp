#ifndef STAN_IO_JSON_RAPIDJSON_PARSER_HPP
#define STAN_IO_JSON_RAPIDJSON_PARSER_HPP


#include "rapidjson/reader.h"

#include <stan/io/validate_zero_buf.hpp>
#include <stan/io/json/json_error.hpp>
#include <stdexcept>
#include <iostream>
#include <istream>
#include <limits>
#include <sstream>
#include <string>


namespace stan {
    namespace json {

        template <typename Handler>
        class rapidjson_parser {
            public:
                rapidjson_parser(Handler& h,
                        std::istream& in)
                : h_(h),
                    in_(in),
                    next_char_(0),
                    line_(0),
                    column_(0)
                {  }

                void parse() {
                    h_.start_text();
                    h_.start_object();
                    h_.key("dada");
                    h_.number_double(4.0);
                    h_.key("baba");
                    h_.number_long(4);
                    h_.end_object();
                    h_.end_text();
                }
            private: 

                Handler& h_;
                std::istream& in_;
                char next_char_;
                size_t line_;
                size_t column_;
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
            rapidjson_parser<Handler>(handler, in).parse();
        }
    }
}
#endif
