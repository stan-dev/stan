#ifndef TEST__UNIT__INSTRUMENTED_CALLBACKS_HPP
#define TEST__UNIT__INSTRUMENTED_CALLBACKS_HPP

#include <stan/callbacks/stream_writer.hpp>
#include <stan/callbacks/writer.hpp>
#include <stan/callbacks/interrupt.hpp>
#include <Eigen/Dense>
#include <map>
#include <string>
#include <iostream>
#include <exception>

namespace stan {
  namespace test {
    namespace unit {

          /**
           * instrumented_interrupt counts the number of times it is
           * called and makes the count accessible via a method.
           */
          class instrumented_interrupt: public stan::callbacks::interrupt {
          public:
            instrumented_interrupt() :
              counter_(0) {}

            void operator()() {counter_++;}

            unsigned int call_count() {return counter_;}

          private:
            unsigned int counter_;
          };

          /**
           * instrumented_writer counts the number of times it is called through
           * each route and makes the count available via methods.
           * Stores all arguments passed and makes them available via
           * methods.
           */
          class instrumented_writer : public stan::callbacks::writer {
          public:
            instrumented_writer() {}

            void operator()(const std::string& key, double value) {
              counter_["string_double"]++;
              string_double.push_back(std::make_pair(key, value));
            }

            void operator()(const std::string& key, int value) {
              counter_["string_int"]++;
              string_int.push_back(std::make_pair(key, value));
            }

            void operator()(const std::string& key, const std::string& value) {
              counter_["string_string"]++;
              string_string.push_back(std::make_pair(key, value));
            }

            void operator()(const std::string& key,
                            const double* values,
                            int n_values) {
              counter_["string_pdouble_int"]++;
              if (n_values == 0) return;
              std::vector<double> value(n_values);
              for (int i = 0; i < n_values; ++i)
                value[i] = values[i];
              string_pdouble_int.push_back(std::make_pair(key, value));
            }

            void operator()(const std::string& key,
                            const double* values,
                            int n_rows, int n_cols) {
              counter_["string_pdouble_int_int"]++;
              if (n_rows == 0 || n_cols == 0) return;
              Eigen::MatrixXd value(n_rows, n_cols);
              for (int i = 0; i < n_rows; ++i)
                for (int j = 0; j < n_cols; ++j)
                  value(i,j) = values[i*n_cols+j];
              string_pdouble_int_int.push_back(std::make_pair(key, value));
            }

            void operator()(const std::vector<std::string>& names) {
              counter_["vector_string"]++;
              vector_string.push_back(names);
            }

            void operator()(const std::vector<double>& state) {
              counter_["vector_double"]++;
              vector_double.push_back(state);
            }

            void operator()() {
              counter_["empty"]++;
            }

            void operator()(const std::string& message) {
              counter_["string"]++;
              string.push_back(message);
            }

            unsigned int call_count() {
              unsigned int n = 0;
              for (std::map<std::string, int>::iterator it=counter_.begin();
                  it!=counter_.end(); ++it)
                n += it->second;
              return n;
            }

            unsigned int call_count(std::string s) {
              return counter_[s];
            }

          std::vector<std::pair<std::string, double> > string_double_values() {
            return string_double;
          };

          std::vector<std::pair<std::string, int> > string_int_values() {
            return string_int;
          };

          std::vector<std::pair<std::string, std::string> > string_string_values() {
            return string_string;
          };

          std::vector<std::pair<std::string, std::vector<double> > > string_pdouble_int_values() {
            return string_pdouble_int;
          };

          std::vector<std::pair<std::string, Eigen::MatrixXd> > string_pdouble_int_int_values() {
            return string_pdouble_int_int;
          };

          std::vector<std::vector<std::string> > vector_string_values() {
            return vector_string;
          };

          std::vector<std::vector<double> > vector_double_values() {
            return vector_double;
          };

          std::vector<std::string> string_values() {
            return string;
          };


          private:
            std::map<std::string, int> counter_;
            std::vector<std::pair<std::string, double> > string_double;
            std::vector<std::pair<std::string, int> > string_int;
            std::vector<std::pair<std::string, std::string> > string_string;
            std::vector<std::pair<std::string, std::vector<double> > > string_pdouble_int;
            std::vector<std::pair<std::string, Eigen::MatrixXd> > string_pdouble_int_int;
            std::vector<std::vector<std::string> > vector_string;
            std::vector<std::vector<double> > vector_double;
            std::vector<std::string> string;


          };

    }
  }
}


#endif
