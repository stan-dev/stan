#include <gtest/gtest.h>
#include <test/test-models/performance/logistic.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <test/performance/utility.hpp>
#include <boost/algorithm/string/trim.hpp>


class performance : public ::testing::Test {
public:
  static void SetUpTestCase() {
    N = 100;
    milliseconds_per_run.resize(N);
    last_draws_per_run.resize(N);
  }

  static int N;
  static Eigen::Matrix<long, -1, 1> milliseconds_per_run;
  static std::vector<Eigen::Matrix<double, -1, 1> > last_draws_per_run;
};

int performance::N;
Eigen::Matrix<long, -1, 1> performance::milliseconds_per_run;
std::vector<Eigen::Matrix<double, -1, 1> > performance::last_draws_per_run;

Eigen::Matrix<double, -1, 1> get_last_iteration_from_file(const char* filename) {
  Eigen::Matrix<double, -1, 1> draw;
  const char comment = '#';
  
  std::ifstream file_stream(filename);
  std::string line;
  std::string last_values;
  while (std::getline(file_stream, line)) {
    if (line.length() > 0 && line[0] != comment)
      last_values = line;
  }
  
  std::stringstream values_stream(last_values);
  std::vector<std::string> values;
  std::string value;
  while (std::getline(values_stream, value, ','))
    values.push_back(value);

  draw.resize(values.size());
  for (int n = 0; n < draw.size(); ++n) {
    draw[n] = atof(values[n].c_str());
  }
  
  return draw;
}


TEST_F(performance, run) {
  const int argc = 11;
  const char* argv[] = {"logistic", 
                        "sample", "num_samples=10000", 
                        "init=0", 
                        "random", "seed=0",
                        "data", "file=src/test/test-models/performance/logistic.data.R",
                        "output", "refresh=10000", "file=test/performance/logistic_output.csv"};
  using boost::posix_time::ptime;
  using boost::posix_time::microsec_clock;

  ptime time_start;
  for (int n = 0; n < N; ++n) {
    time_start = microsec_clock::universal_time(); // start timer
    stan::common::command<stan_model>(argc, argv);
    milliseconds_per_run[n] 
      = (microsec_clock::universal_time() - time_start).total_milliseconds();
    last_draws_per_run[n] = get_last_iteration_from_file("test/performance/logistic_output.csv");
  }
  SUCCEED();
}

// evaluate
TEST_F(performance, values_from_tagged_version) {
  int N_values = 8;
  ASSERT_EQ(N_values, last_draws_per_run[0].size())
    << "last tagged version, 2.5.0, had " << N_values << " elements";

  Eigen::Matrix<double, -1, 1> first_run = last_draws_per_run[0];
  EXPECT_FLOAT_EQ(-67.5276, first_run[0])
    << "lp__: index 0";
  EXPECT_FLOAT_EQ(0.773672, first_run[1])
    << "accept_stat__: index 1";
  EXPECT_FLOAT_EQ(0.901013, first_run[2])
    << "stepsize__: index 2";
  EXPECT_FLOAT_EQ(2, first_run[3])
    << "treedepth__: index 3";
  EXPECT_FLOAT_EQ(3, first_run[4])
    << "n_leapfrog__: index 4";
  EXPECT_FLOAT_EQ(0, first_run[5])
    << "n_divergent__: index 5";
  EXPECT_FLOAT_EQ(1.71115, first_run[6])
    << "beta.1: index 6";
  EXPECT_FLOAT_EQ(-0.291085, first_run[7])
    << "beta.2: index 7";
}

TEST_F(performance, values_same_run_to_run) {
  int N_values = last_draws_per_run[0].size();
  
  for (int i = 0; i < N_values; i++) {
    double expected_value = last_draws_per_run[0][i];
    for (int n = 1; n < N; n++) {
      EXPECT_FLOAT_EQ(expected_value, last_draws_per_run[n][i])
        << "expecting run to run values to be the same. Found run "
        << n << " to have different values than the 0th run for "
        << "index: " << i;
    }
  }
}

template <typename T>
std::string quote(const T& val) {
  std::stringstream quoted_val;
  quoted_val << "\""
             << val
             << "\"";
  return quoted_val.str();
}
 
std::string get_git_hash() {
  run_command_output git_hash = run_command("git rev-parse HEAD");
  if (git_hash.hasError)
    return "NA";
  boost::trim(git_hash.body);
  return git_hash.body;
}

std::string get_git_date() {
  run_command_output git_date = run_command("git log --format=%ct -1");
  if (git_date.hasError)
    return "NA";
  boost::trim(git_date.body);
  
  long timestamp = atol(git_date.body.c_str());
  std::time_t date(timestamp);

  return to_iso_extended_string(boost::posix_time::from_time_t(date));
}

TEST_F(performance, write_results_to_disk) {
  using boost::posix_time::second_clock;
  
  std::stringstream header;
  std::stringstream line;

  // current date / time
  header << quote("date");
  line << quote(to_iso_extended_string(second_clock::universal_time()));
  
  // git hash
  header << "," << quote("git hash") << "," << quote("git date");
  line << "," << quote(get_git_hash()) << "," << quote(get_git_date());

  // model name: "logistic"
  header << "," << quote("model name");
  line << "," << quote("logistic");

  // N times
  for (int n = 0; n < N; n++) {
    std::stringstream ss;
    ss << "run " << n+1;
    header << "," << quote(ss.str());
    line << "," << milliseconds_per_run[n];
  }

  // append output to: test/performance/performance.csv
  bool write_header = false;
  std::fstream file_stream;
  
  file_stream.open("test/performance/performance.csv", 
                   std::ios_base::in);
  if (file_stream.peek() == std::fstream::traits_type::eof()) {
    write_header = true;
  } else {
    std::string file_header;
    std::getline(file_stream, file_header);
    std::cout << "existing file header: " << file_header << std::endl;
    
    EXPECT_EQ(file_header, header.str()) 
      << "header of file is different";
    if (file_header != header.str())
      write_header = true;
  }
  file_stream.close();

  file_stream.open("test/performance/performance.csv",
                   std::ios_base::app);
  if (write_header)
    file_stream << header.str() << std::endl;
  file_stream << line.str() << std::endl;
  file_stream.close();


}
