#ifndef STAN_INTERFACE_CALLBACKS_WRITER_PSQL_WRITER_HPP
#define STAN_INTERFACE_CALLBACKS_WRITER_PSQL_WRITER_HPP

#include <stan/interface_callbacks/writer/base_writer.hpp>
#include <stan/interface_callbacks/writer/psql_writer_helpers.hpp>
#include <pqxx/pqxx>
#include <ostream>
#include <vector>
#include <queue>
#include <string>
#include <thread>


namespace stan {
  namespace interface_callbacks {
    namespace writer {

      /**
       * psql_writer writes to the postgres database specific by the URI
       * string.  Since it relies on the standard postgres library, the
       * URI lets you rely on any of the usual authentication methods
       * for psql.  For local connections this doesn't matter but the
       * goal is to make remote writes doable and reliable so SSL/TLS
       * client/server authentication and encryption are in the scope of
       * the writer.
       *
       * The database side strategy is to identify each run with a
       * (short) hash that gets written with each key/value/index/type
       * message.  
       *
       * The tables we use: lower complexity, avoid dealing with binary
       * writes for vector/matrix, all key_value in more messy key_value
       * table.  
       *  - runs
       *  - key_value (as hash, key, idx, row, column, double, string, int)
       *  - parameter_names
       *  - parameter_samples
       *  - messages
       *
       *
       * Each call to a simple (single-value) method is done with the
       * main thread.  Initially, each call to a multi-value method is
       * done with the main thread.  Eventually it might be worthwhile
       * to send the work to a separate thread if the vector is very
       * long so that the write can happen simultaneously with the
       * numerical calculations (and hopefully be done before the next
       * set is sent over).  With a slow connection there is potential
       * for work to pile up so either waiting in the writer or
       * something fancier might be required.  Not going to think about
       * that initially.
       */
      class psql_writer : public base_writer {
      public:
        /**
         * Constructor.
         *
         * @param uri std::string passed to the psql library to establish a
         *   connection. 
         * @param id std::string passed as a user-generated tag for the
         *   runs table.  Not used as an index internally. 
         */
        psql_writer(const std::string& uri = "", const std::string id = ""):
            uri__(uri), id__(id), iteration__(0) {

          conn__ = new pqxx::connection(uri);
          conn__->perform(do_sql(create_runs_sql));
          conn__->perform(do_sql(create_key_value_sql));
          conn__->perform(do_sql(create_parameter_names_sql));
          conn__->perform(do_sql(create_parameter_samples_sql));
          conn__->perform(do_sql(create_messages_sql));
          
          pqxx::work write(*conn__, "run_write");
          pqxx::result runs_result = write.exec("INSERT INTO runs (id) VALUES ('" + id__ + "') RETURNING hash;");
          hash__ = runs_result[0][0].c_str();   
          write.commit();

          conn__->prepare("write_key_value", write_key_value_sql);
          conn__->prepare("write_parameter_name", write_parameter_name_sql);
          conn__->prepare("write_parameter_sample", write_parameter_sample_sql);
          conn__->prepare("write_message", write_message_sql);
          
         
        }

        ~psql_writer() {
          conn__->disconnect();
          delete conn__;
        }

        void operator()(const std::string& key, double value) {
          conn__->perform(write_key_double(hash__, key, value));
        }

        void operator()(const std::string& key, int value) {
          conn__->perform(write_key_integer(hash__, key, value));
          
        }

        void operator()(const std::string& key, const std::string& value) {
          conn__->perform(write_key_string(hash__, key, value));
        }

        void operator()(const std::string& key,
                        const double* values,
                        int n_values
        ) {
          conn__->perform(write_key_doubles_n(hash__, key, values, n_values));
        }

        void operator()(const std::string& key, const double* values,
                        int n_rows, int n_cols) {
          conn__->perform(write_key_doubles_rows_columns(hash__, key, values, n_rows, n_cols));
        }

        void operator()(const std::vector<std::string>& names) {
          names__ = names;
          conn__->perform(write_parameter_names(hash__, names));
        }

        void operator()(const std::vector<double>& state) {
          ++iteration__;
          if (state.size() < 1000) {
            conn__->perform(write_parameter_samples(hash__, iteration__, names__, state));
          } else {
            std::thread write_thread(&psql_writer::threadable_sample_write, this, uri__, hash__, iteration__, names__, state);
            write_thread.detach();
          }
        }

        void operator()() { }

        void operator()(const std::string& message) {
          conn__->perform(write_message(hash__, message));
        }

        void threadable_sample_write(std::string uri, std::string hash, int iteration, 
          std::vector<std::string> names, std::vector<double> state){
            pqxx::connection conn(uri);
            conn.prepare("write_parameter_sample", write_parameter_sample_sql);
            conn.perform(write_parameter_samples(hash, iteration, names, state));
            conn.disconnect();
        }

      private:
        pqxx::connection* conn__;
        std::vector<std::string> names__;
        std::string uri__;
        std::string hash__;
        std::string id__;
        int iteration__;

        static const std::string create_runs_sql;
        static const std::string create_key_value_sql;
        static const std::string create_parameter_names_sql;
        static const std::string create_parameter_samples_sql;
        static const std::string create_messages_sql;

        static const std::string write_key_value_sql;
        static const std::string write_parameter_name_sql;
        static const std::string write_parameter_sample_sql;
        static const std::string write_message_sql;
      };

      const std::string psql_writer::create_runs_sql = "CREATE TABLE IF NOT EXISTS "
        "runs("
        "hash SERIAL PRIMARY KEY,"
        "timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,"
        "id VARCHAR(200) NOT NULL"
      ");";
      const std::string psql_writer::create_key_value_sql = "CREATE TABLE IF NOT EXISTS "
        "key_value("
        "row_id SERIAL PRIMARY KEY,"
        "hash INT REFERENCES runs,"
        "key VARCHAR(50),"
        "idx INTEGER,"
        "row_idx INTEGER,"
        "col_idx INTEGER,"
        "double DOUBLE PRECISION,"
        "string VARCHAR(300),"
        "integer INTEGER"
      ");";
      const std::string psql_writer::create_parameter_names_sql = "CREATE TABLE IF NOT EXISTS "
        "parameter_names("
        "row_id BIGSERIAL PRIMARY KEY,"
        "hash INT REFERENCES runs,"
        "name VARCHAR(200)"
      ");";
      const std::string psql_writer::create_parameter_samples_sql = "CREATE TABLE IF NOT EXISTS "
        "parameter_samples("
        "row_id BIGSERIAL PRIMARY KEY, " 
        "hash INT REFERENCES runs, "
        "iteration INTEGER, "
        "name VARCHAR(200), "
        "value DOUBLE PRECISION"
      ");";
      const std::string psql_writer::create_messages_sql = "CREATE TABLE IF NOT EXISTS "
        "messages("
        "row_id BIGSERIAL PRIMARY KEY,"
        "hash INT REFERENCES runs,"
        "message VARCHAR(200)"
      ");";

      const std::string psql_writer::write_key_value_sql = "INSERT INTO key_value "
        "(hash, key, idx, row_idx, col_idx, double, string, integer)"
        " VALUES "
        "($1, $2, $3, $4, $5, $6, $7, $8);";
      const std::string psql_writer::write_parameter_name_sql = "INSERT INTO parameter_names "
        "(hash, name)"
        " VALUES "
        "($1, $2);";
      const std::string psql_writer::write_parameter_sample_sql = "INSERT INTO parameter_samples "
        "(hash, iteration, name, value)"
        " VALUES "
        "($1, $2, $3, $4);";
      const std::string psql_writer::write_message_sql = "INSERT INTO messages "
        "(hash, message)"
        " VALUES "
        "($1, $2);";

    }
  }
}

#endif
