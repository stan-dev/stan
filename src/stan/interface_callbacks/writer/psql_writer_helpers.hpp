#ifndef STAN_INTERFACE_CALLBACKS_WRITER_PSQL_WRITER_HELPERS_HPP
#define STAN_INTERFACE_CALLBACKS_WRITER_PSQL_WRITER_HELPERS_HPP

#include <pqxx/pqxx>
#include <vector>
#include <string>

namespace stan {
  namespace interface_callbacks {
    namespace writer {

      class do_sql : public pqxx::transactor<>
      {
        const std::string sql__;
      
      public:
        explicit do_sql(const std::string sql) :
          pqxx::transactor<>("do_sql"), sql__(sql) { }
      
        void operator()(pqxx::work &T)
        {
          T.exec(sql__);
        }
      };

      class write_key_double : public pqxx::transactor<>
      {
        const std::string hash__;
        const std::string key__;
        const double value__;
      
      public:
        explicit write_key_double(const std::string hash, const std::string key, 
          const double value) : pqxx::transactor<>("write_key_double"), 
          hash__(hash), key__(key), value__(value) { }
      
        void operator()(pqxx::work &T)
        {
          T.prepared("write_key_value")(hash__)(key__)()()()(value__)()().exec();
        }
      };

      class write_key_string : public pqxx::transactor<>
      {
        const std::string hash__;
        const std::string key__;
        const std::string value__;
      
      public:
        explicit write_key_string(const std::string hash, const std::string key, 
          const std::string value) : pqxx::transactor<>("write_key_string"), 
          hash__(hash), key__(key), value__(value) { }
      
        void operator()(pqxx::work &T)
        {
          T.prepared("write_key_value")(hash__)(key__)()()()()(value__)().exec();
        }
      };

      class write_key_integer : public pqxx::transactor<>
      {
        const std::string hash__;
        const std::string key__;
        const int value__;
      
      public:
        explicit write_key_integer(const std::string hash, 
          const std::string key, int value) : 
          pqxx::transactor<>("write_key_integer"), 
          hash__(hash), key__(key), value__(value) { }
      
        void operator()(pqxx::work &T)
        {
          T.prepared("write_key_value")(hash__)(key__)()()()()()(value__).exec();
        }
      };

      class write_key_doubles_n : public pqxx::transactor<>
      {
        const std::string hash__;
        const std::string key__;
        const double* value__;
        int n_values__;
      
      public:
        explicit write_key_doubles_n(const std::string hash, 
          const std::string key, const double* value, int n_values) :
          pqxx::transactor<>("write_key_doubles_n"), 
          hash__(hash), key__(key), value__(value), n_values__(n_values) { }
      
        void operator()(pqxx::work &T)
        {
          for (unsigned int i=0; i < n_values__; ++i) {
            T.prepared("write_key_value")(hash__)(key__)(i)()()(value__[i])()().exec();
          }
        }
      };

      class write_key_doubles_rows_columns : public pqxx::transactor<>
      {
        const std::string hash__;
        const std::string key__;
        const double* value__;
        int n_rows__;
        int n_cols__;
      
      public:
        explicit write_key_doubles_rows_columns(const std::string hash, 
          const std::string key, const double* value, int n_rows, int n_cols) :
          pqxx::transactor<>("write_key_doubles_rows_columns"), 
          hash__(hash), key__(key), value__(value), n_rows__(n_rows), n_cols__(n_cols) { }
      
        void operator()(pqxx::work &T)
        {
          for (int i=0; i < n_rows__; ++i) {
            for (int j = 0; j < n_cols__; ++j) {
              T.prepared("write_key_value")(hash__)(key__)()(i)(j)(value__[i*n_cols__+j])()().exec();
            }
          }
        }
      };

      class write_parameter_names : public pqxx::transactor<>
      {
        const std::string hash__;
        const std::vector<std::string> names__;
      
      public:
        explicit write_parameter_names(const std::string hash, 
          const std::vector<std::string> names) : 
          pqxx::transactor<>("write_parameter_names"), 
          hash__(hash), names__(names) { }
      
        void operator()(pqxx::work &T)
        {
          for (unsigned int i = 0; i < names__.size(); ++i) {
            T.prepared("write_parameter_name")(hash__)(names__[i]).exec();
          }
        }
      };

      class write_parameter_samples : public pqxx::transactor<>
      {

        const std::string hash__;
        const double iteration__;
        const std::vector<std::string>& names__;
        const std::vector<double>& values__;
      
      public:
        explicit write_parameter_samples(const std::string hash, 
          const double iteration,
          const std::vector<std::string>& names,
          const std::vector<double>& values) :
          pqxx::transactor<>("write_parameter_samples"), 
          hash__(hash), iteration__(iteration), names__(names), values__(values) { }
      
        void operator()(pqxx::work &T)
        { 
          if (values__.size() != names__.size())
            throw std::range_error("Number of parameter names and values do not match.");
          for (unsigned int i = 0; i < values__.size(); ++i) {
            T.prepared("write_parameter_sample")(hash__)(iteration__)(names__[i])(values__[i]).exec();
          }
        }
       };

      class write_message : public pqxx::transactor<>
      {
        const std::string hash__;
        const std::string message__;
      
      public:
        explicit write_message(const std::string hash, 
          const std::string message) : pqxx::transactor<>("write_message"), 
          hash__(hash), message__(message) { }
      
        void operator()(pqxx::work &T)
        {
          T.prepared("write_message")(hash__)(message__).exec();
        }
      };

    }
  }
}

#endif
