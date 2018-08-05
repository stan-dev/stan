/**
 * Performance test: logistic.
 *
 * This test runs a simple logistic model 100 times with a fixed
 * seed. The test records to a csv file information about the current
 * build, whether the values match the expected known values, whether
 * values are identical between runs, and the runtime in seconds for
 * each of the 100 individual runs.
 *
 * The output file is a csv file located at
 * test/performance/performance.csv.  If the file is empty, a header
 * line will be written.  If the file is not empty and the header line
 * of the file matches, a data row will be appended.  If the file is
 * not empty and the header line is different, a header row and a data
 * row will be appended.
 *
 * Below are the test routines in this test.
 * 1. run: Runs the model 100 times. This test should always
 *    succeed. Test failure indicates that the model could not be run.
 * 2. values_from_tagged_version: Compares the first run's final
 *    iteration against known values. Test will succeed if the values
 *    match the known values; the test will fail otherwise.
 * 3. values_same_run_to_run: Compares the first run's final iteration
 *    against all runs' final iterations. Test will succeed if all the
 *    values match; the test will fail otherwise.
 * 4. write_results_to_disk: Writes the results to disk. Test will
 *    succeed if writing to an empty file or to one with a matching
 *    header; the test will fail if the header in the existing file
 *    does not match the current header.
 *
 * Note: the values_from_tagged_version will need to be updated
 * if Stan's RNG changes, the seeding mechanism changes, or the
 * algorithm changes. The other routines should be fine.
 *
 * Each run of this test produces a single, csv row with 106 columns
 * in the output file:
 *   1. current date. Formatted using std::ctime: "Www Mmm dd hh:mm:ss yyyy"
 *   2. git hash. The current 40-character SHA-1 git hash or "NA".
 *   3. git date. The date of the last commit or "NA".
 *   4. model name. "logistic"
 *   5. matches tagged version. Either "yes" or "no".
 *   6. all values the same. Either "yes" or "no".
 *   7--106. run time (in seconds).
 */

#include <gtest/gtest.h>
#include <test/test-models/performance/logistic.hpp>
#include <stan/version.hpp>
#include <ctime>
#include <test/performance/utility.hpp>
#include <boost/algorithm/string/trim.hpp>


class performance : public ::testing::Test {
public:
  static void SetUpTestCase() {
    N = 100;
    seconds_per_run.resize(N);
    last_draws_per_run.resize(N);
    matches_tagged_version = false;
    all_values_same = false;
  }

  static int N;
  static std::vector<double> seconds_per_run;
  static std::vector<std::vector<double> > last_draws_per_run;
  static bool matches_tagged_version;
  static bool all_values_same;
};

int performance::N;
std::vector<double> performance::seconds_per_run;
std::vector<std::vector<double> > performance::last_draws_per_run;
bool performance::matches_tagged_version;
bool performance::all_values_same;


using stan::test::performance::run_command_output;
using stan::test::performance::run_command;
using stan::test::performance::get_last_iteration_from_file;
using stan::test::performance::quote;
using stan::test::performance::get_git_hash;
using stan::test::performance::get_git_date;
using stan::test::performance::get_date;

TEST_F(performance, run) {
  clock_t t;
  for (int n = 0; n < N; ++n) {
    std::cout << "iteration: " << n << " / " << N << std::endl;
    t = clock();      // start timer
    stan::test::performance::command<stan_model>(1000,
                                                 10000,
                                                 "src/test/test-models/performance/logistic.data.R",
                                                 "test/performance/logistic_output.csv",
                                                 0U);
    t = clock() - t;  // end timer
    seconds_per_run[n] = static_cast<double>(t) / CLOCKS_PER_SEC;
    last_draws_per_run[n]
      = get_last_iteration_from_file("test/performance/logistic_output.csv");
  }
  SUCCEED();
}

// evaluate
TEST_F(performance, values_from_tagged_version) {
  int N_values = 9;
  ASSERT_EQ(N_values, last_draws_per_run[0].size())
    << "last tagged version, 2.17.0, had " << N_values << " elements";

  std::vector<double> first_run = last_draws_per_run[0];
  EXPECT_FLOAT_EQ(-65.781998, first_run[0])
    << "lp__: index 0";

  EXPECT_FLOAT_EQ(1.0, first_run[1])
    << "accept_stat__: index 1";

  EXPECT_FLOAT_EQ(0.76853198, first_run[2])
    << "stepsize__: index 2";

  EXPECT_FLOAT_EQ(2, first_run[3])
    << "treedepth__: index 3";

  EXPECT_FLOAT_EQ(7, first_run[4])
    << "n_leapfrog__: index 4";

  EXPECT_FLOAT_EQ(0, first_run[5])
    << "divergent__: index 5";

  EXPECT_FLOAT_EQ(66.6695, first_run[6])
    << "energy__: index 6";

  EXPECT_FLOAT_EQ(1.55186, first_run[7])
    << "beta.1: index 7";

  EXPECT_FLOAT_EQ(-0.52400702, first_run[8])
    << "beta.2: index 8";

  matches_tagged_version = !HasNonfatalFailure();
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
  all_values_same = !HasNonfatalFailure();
}

TEST_F(performance, check_output_is_same) {
  std::ifstream file_stream;
  file_stream.open("test/performance/logistic_output.csv",
                   std::ios_base::in);
  ASSERT_TRUE(file_stream.good());

  std::string line, expected;

  getline(file_stream, line);
  ASSERT_EQ("lp__,accept_stat__,stepsize__,treedepth__,n_leapfrog__,divergent__,energy__,beta.1,beta.2", line);
  ASSERT_TRUE(file_stream.good());

  getline(file_stream, line);
  ASSERT_EQ("# Adaptation terminated", line);
  ASSERT_TRUE(file_stream.good());

  file_stream.close();
}

TEST_F(performance, write_results_to_disk) {
  std::stringstream header;
  std::stringstream line;

  // current date / time
  header << quote("date");
  line << quote(get_date());

  // git hash
  header << "," << quote("git hash") << "," << quote("git date");
  line << "," << quote(get_git_hash()) << "," << quote(get_git_date());

  // model name: "logistic"
  header << "," << quote("model name");
  line << "," << quote("logistic");

  // matches tagged values
  header << "," << quote("matches tagged version");
  line << "," << quote(matches_tagged_version ? "yes" : "no");

  // all values same
  header << "," << quote("all values same");
  line << "," << quote(all_values_same ? "yes" : "no");

  // N times
  for (int n = 0; n < N; n++) {
    std::stringstream ss;
    ss << "run " << n+1;
    header << "," << quote(ss.str());
    line << "," << seconds_per_run[n];
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
