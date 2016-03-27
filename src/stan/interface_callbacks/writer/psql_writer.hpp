#ifndef STAN_INTERFACE_CALLBACKS_WRITER_PSQL_WRITER_HPP
#define STAN_INTERFACE_CALLBACKS_WRITER_PSQL_WRITER_HPP

#include <stan/interface_callbacks/writer/base_writer.hpp>
#include <psql_writer_helpers.hpp>
#include <pqxx/pqxx>
#include <ostream>
#include <vector>
#include <string>


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
       * The data format strategy in general is to write everything as
       * key/value/index/type pairs. The simple tables required would
       * be:
       *  - runs
       *  - key_value_double
       *  - key_value_integer
       *  - key_value_string
       *  - key_vector
       *  - key_matrix
       *  - parameter_names
       *  - parameter_samples
       *  - messages
       *
       * The tables we use instead are: lower complexity, avoid dealing with binary
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
            uri__(uri), id__(id) {
          conn__ = new pqxx::connection(uri);
          conn__.perform(do_sql(create_runs_sql));
          conn__.perform(do_sql(create_key_value_sql));
          conn__.perform(do_sql(create_parameter_name_sql));
          conn__.perform(do_sql(create_parameter_sample_sql));
          conn__.perform(do_sql(create_message_sql));
          
          pqxx::work write(conn__, "run_write");
          pqxx::result runs_result = write.exec("INSERT INTO runs (id) VALUES (" + id__ + ") RETURNING hash;");
          hash__ = runs_result[0][0].c_str();   
          write.commit();

          conn__.prepare("write_key_value", write_key_value_sql);
          conn__.prepare("write_parameter_name", write_parameter_name_sql);
          conn__.prepare("write_parameter_sample", write_parameter_sample_sql);
          conn__.prepare("write_message", write_message_sql);

        }

        ~psql_writer() {
          delete conn__;
        }

        void operator()(const std::string& key, double value) {
          conn__.perform(write_key_double(hash__, key, value));
        }

        void operator()(const std::string& key, int value) {
          conn__.perform(write_key_integer(hash__, key, value));
          
        }

        void operator()(const std::string& key, const std::string& value) {
          conn__.perform(write_key_string(hash__, key, value));
        }

        void operator()(const std::string& key,
                        const double* values,
                        int n_values
        ) {
          conn__.perform(write_key_doubles_n(hash__, key, values, n_values));
        }

        void operator()(const std::string& key, const double* values,
                        int n_rows, int n_cols) {
          conn__.perform(write_key_doubles_rows_columns(hash__, key, values, n_rows, n_cols));
        }

        void operator()(const std::vector<std::string>& names) {
          names__ = names;
          conn__.perform(write_parameter_names(hash__, names));
        }

        void operator()(const std::vector<double>& state) {
          conn__.perform(write_parameter_samples(hash__, names__, state));
        }

        void operator()() { }

        void operator()(const std::string& message) {
          conn__.perform(write_messages(hash__, message));
        }

      private:
        pqxx::connection conn__;
        std::vector<string> names__;
        std::string hash__;
        std::string id__;

        static const string create_runs_sql;
        static const string create_key_value_sql;
        static const string create_parameter_names_sql;
        static const string create_parameter_samples_sql;
        static const string create_messages_sql;

        static const string write_key_value_sql;
        static const string write_parameter_names_sql;
        static const string write_parameter_samples_sql;
        static const string write_messages_sql;
      };

      psql_writer::create_runs_sql = "CREATE TABLE IF NOT EXISTS "
        "runs("
        "hash SERIAL PRIMARY KEY,"
        "timestamp TIMESTAMP WITH TIMEZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,"
        "id VARCHAR(200) NOT NULL"
      ");";
      psql_writer::create_key_value_sql = "CREATE TABLE IF NOT EXISTS "
        "key_value("
        "row_id SERIAL PRIMARY KEY,"
        "hash REFERENCES runs,"
        "key VARCHAR(50),"
        "idx INTEGER,"
        "row INTEGER,"
        "column INTEGER,"
        "double DOUBLE PRECISION,"
        "string VARCHAR(300),"
        "integer INTEGER"
      ");";
      psql_writer::create_parameter_names_sql = "CREATE TABLE IF NOT EXISTS "
        "parameter_names("
        "row_id BIGSERIAL PRIMARY KEY,"
        "hash REFERENCES runs,"
        "names VARCHAR(200),"
      ");";
      psql_writer::create_parameter_samples_sql = "CREATE TABLE IF NOT EXISTS "
        "parameter_samples("
        "row_id BIGSERIAL PRIMARY KEY,"
        "hash REFERENCES runs,"
        "iteration INTEGER,"
        "value DOUBLE PRECISION,"
      ");";
      psql_writer::create_messages_sql = "CREATE TABLE IF NOT EXISTS "
        "messages("
        "row_id BIGSERIAL PRIMARY KEY,"
        "hash REFERENCES runs,"
        "message VARCHAR(200),"
      ");";

      psql_writer::write_key_value_sql = "INSERT INTO key_value "
        "(hash, key, idx, row, column, double, string, integer)"
        " VALUES "
        "($1, $2, $3, $4, $5, $6, $7, $8);";
      psql_writer::write_parameter_name_sql = "INSERT INTO parameter_names "
        "(hash, names)"
        " VALUES "
        "($1, $2);";
      psql_writer::write_parameter_sample_sql = "INSERT INTO parameter_samples "
        "(hash, iteration, value)"
        " VALUES "
        "($1, $2, $3);";
      psql_writer::write_message_sql = "INSERT INTO messages "
        "(hash, message)"
        " VALUES "
        "($1, $2);";

    }
  }
}

#endif
