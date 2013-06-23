#include <gtest/gtest.h>
#include <test/models/utility.hpp>
#include <fstream>
#include <algorithm>
#include <stan/io/dump.hpp>
#include <stan/io/csv_writer.hpp>
#include <stan/mcmc/chains.hpp>
#include <boost/math/distributions/students_t.hpp>
#include <boost/math/distributions/binomial.hpp>


class TestInfo {
public:
  size_t N;
  size_t M;
  size_t iterations;
  
  TestInfo(size_t N, size_t M, size_t iterations) 
    : N(N), M(M), iterations(iterations) { }
};

std::vector<TestInfo> getTestCases() {
  std::vector<TestInfo> testCases;
  
  testCases.push_back(TestInfo(128,  2, 2000));
  testCases.push_back(TestInfo(1024, 2, 2000));
  testCases.push_back(TestInfo(4096, 2, 2000));
  
  /*
  testCases.push_back(TestInfo(128,  8, 2000));
  testCases.push_back(TestInfo(1024, 8, 2000));
  testCases.push_back(TestInfo(4096, 8, 2000));
  
  testCases.push_back(TestInfo(128,  32, 300));
  testCases.push_back(TestInfo(1024, 32, 300));
  testCases.push_back(TestInfo(4096, 32, 300));
  
  testCases.push_back(TestInfo(1024, 128, 1000));
  testCases.push_back(TestInfo(4096, 128, 1000));
  
  testCases.push_back(TestInfo(1024, 512, 100000));
  testCases.push_back(TestInfo(4096, 512, 100000));*/
  
  return testCases;
}

::std::ostream& operator<<(::std::ostream& os, const TestInfo& info) {
  os << "TestInfo:" << std::endl
     << "\tN:          " << info.N << std::endl
     << "\tM:          " << info.M << std::endl
     << "\titerations: " << info.iterations << std::endl;
  return os;  // whatever needed to print bar to os
}


class LogisticSpeedTest :
  public testing::TestWithParam<TestInfo> {
public:
  static const size_t num_chains;
  static bool has_R;
  static bool has_jags;
  static std::string path;
  static std::ofstream output_file;
  static std::string Rscript;
  static std::vector<std::string> data_files;
  static size_t max_M;

  static void SetUpTestCase() {
    std::vector<std::string> model_path;
    model_path.push_back("models");
    model_path.push_back("speed");
    model_path.push_back("logistic");

    path = convert_model_path(model_path);

    Rscript = "logistic_generate_data.R";

    std::vector<TestInfo> testCases = getTestCases();
    max_M = 0;
    for (size_t i = 0; i < testCases.size(); i++) {
      TestInfo info = testCases[i];
      std::stringstream filename;
      filename << "logistic_" << info.N << "_" << info.M;
      max_M = info.M > max_M ? info.M : max_M;
      data_files.push_back(filename.str());
    }
    
    
    std::stringstream output_filename;
    output_filename << path << get_path_separator() 
                    << "logistic.csv";
    {
      std::string tmp(output_filename.str());
      output_file.open(tmp.c_str());
    }
    output_file << "Program,"
                << "N,"
                << "M,"
                << "milliseconds,"
                << "Min effective samples,"
                << "ms per min effective samples";
    for (size_t m = 0; m < max_M; m++) {
      output_file << "," << "effective sample size " << m;
    }
    output_file << "\n";
    //    - Info about the run: 'Stan', n, m
    //    - time
    //    - min effective samples
    //    - time / min effective samples
    //    - effective samples 1..m
  }
  
  static void TearDownTestCase() {
    output_file.close();
  }

  /** 
   * Executes the Stan model and returns elapsed time.
   *
   * The Stan model is executed <code>num_chains</code> times.
   * The <code>command</code> argument has the basic Stan command
   * to run. The <code>filename</code> argument has the basename
   * for the output samples. This is append with the chain id and
   * the suffix '.csv'.
   *
   * The standard output stream and error output stream for each 
   * chain is recorded and output in command_outputs.
   * 
   * 
   * @param[in] command The command to run.
   * @param[in] filename The output filename without a suffix.
   * @param[out] command_outputs The captured output per chain.
   * 
   * @return Elapsed time running the commands in milliseconds.
   */
  long run_stan(const std::string& command, const std::string& filename, std::vector<std::string> command_outputs) {
    long time = 0;
    for (size_t chain = 1; chain <= num_chains; chain++) {
      std::stringstream command_chain;
      command_chain << command;
      command_chain << " --chain_id=" << chain
                    << " --samples=" << path << get_path_separator() 
                    << filename << ".chain_" << chain << ".csv";
      std::string command_output;
      try {
        command_output = run_command(command_chain.str(), time);
      } catch(...) {
        ADD_FAILURE() << "Failed running command: " << command_chain.str();
      }
      command_outputs.push_back(command_output);
    }
    return time;
  }

  /** 
   * Creates a chain object based on the filename provided.
   *
   * @param filename base filename.
   * 
   * @return a chain object
   */
  stan::mcmc::chains<> create_chains(const std::string& filename) {
    std::stringstream samples;
    samples << path << get_path_separator()
            << filename << ".chain_1.csv";
  


    std::ifstream ifstream; 
    {
      std::string tmp(samples.str());
      ifstream.open(tmp.c_str());
    }
    stan::io::stan_csv stan_csv = stan::io::stan_csv_reader::parse(ifstream);
    ifstream.close();
    
    stan::mcmc::chains<> chains(stan_csv);
    for (size_t chain = 1; chain < num_chains; chain++) {
      samples.str("");
      samples << path << get_path_separator()
              << filename << ".chain_" << chain << ".csv";
      {
        std::string tmp(samples.str());
        ifstream.open(tmp.c_str());
      }
      stan_csv = stan::io::stan_csv_reader::parse(ifstream);
      ifstream.close();
      
      chains.add(stan_csv);
    }
    return chains;
  }

  /** 
   * Gets the beta from the generated data.
   * 
   * @param[in] filename filename
   * @param[out] beta output vector
   */
  void get_beta(const std::string& filename, std::vector<double>& beta) {
    std::stringstream param_filename;
    param_filename << path << get_path_separator() << filename
                   << "_param.data.R";
    std::string tmp(param_filename.str());
    std::ifstream param_ifstream(tmp.c_str());
    stan::io::dump param_values(param_ifstream);
    
    beta = param_values.vals_r("beta");
  }
  
  /** 
   * Runs the test case.
   * 
   * 1) Get the generated beta.
   * 2) Run Stan num_chains times
   * 3) Test values of sampled parameters
   * 4) Output useful values.
   * 
   * @param filename 
   * @param iterations 
   */
  void test_logistic_speed_stan(const std::string& filename, 
        const size_t iterations,
                                const TestInfo& info) {
    if (!has_R)
      return;
    using std::vector;
    using boost::math::students_t;
    using boost::math::binomial;
    using boost::math::quantile;
    using std::setw;
    
    // 1) Get the generated beta.
    vector<double> beta;
    get_beta(filename, beta);

    // 2) Run Stan num_chains times
    std::stringstream command;
    command << path << get_path_separator() << "logistic"
            << " --data=" << path << get_path_separator() << filename << ".data.R"
            << " --iter=" << iterations
            << " --refresh=" << iterations;
    vector<std::string> command_outputs;  
    long time = run_stan(command.str(), filename, command_outputs);


    // 3) Test values of sampled parameters
    stan::mcmc::chains<> chains = create_chains(filename);
    int num_failed = 0;
    std::stringstream err_message;
    double alpha = 0.05;
    for (size_t index = 0; index < beta.size(); index++) {
      double neff = chains.effective_sample_size(index);
      double sample_mean = chains.mean(index);
      double se = chains.sd(index) / sqrt(neff);
      double z = quantile(students_t(neff-1.0), 1 - alpha/2.0);
      
      if (fabs(beta[index] - sample_mean) > z*se) {
        num_failed++;
        // want the error message to have which, what, how
        err_message << "beta[" << index << "]:"
                    << "\n\texpected:    " << setw(10) << beta[index]
                    << "\n\tsampled:     " << setw(10) << sample_mean
                    << "\n\tneff:        " << setw(10) << neff
                    << "\n\tsplit R.hat: " << setw(10) << chains.split_potential_scale_reduction(index)
                    << "\n\tz:           " << setw(10) << z
                    << "\n\tse:          " << setw(10) << se
                    << "\n\n\tfabs(diff) > z * se: " 
                    << fabs(beta[index] - sample_mean) << " > " << z*se << "\n\n";
      }
    }
    /*double p = 1 - cdf(binomial(beta.size(), alpha), num_failed);
    // this test should fail less than 0.1% of the time.
    if (p < 0.001) {
      EXPECT_EQ(0, num_failed)
        << "Failed " << num_failed << " of " << beta.size() << " comparisons\n"
        << "p: " << p << std::endl
        << "------------------------------------------------------------\n"
        << err_message.str() << std::endl
        << "------------------------------------------------------------\n";
        }*/

    // 4) Output useful values.
    vector<double> neff(beta.size(), 0.0);
    for (size_t m = 0; m < beta.size(); m++) {
      neff[m] = chains.effective_sample_size(m);
    }
    double min_neff = std::min(neff.front(), neff.back());


    //    - Info about the run: 'Stan', n, m
    stan::io::csv_writer writer(output_file);
    writer.write("Stan");
    writer.write((double)info.N);
    writer.write((double)info.M);
    //    - time
    writer.write((double)time);
    //    - min effective samples
    writer.write(min_neff);
    //    - time / min effective samples
    writer.write((double)time / min_neff);
    //    - effective samples 1..m
    for (size_t m = 0; m < neff.size(); m++)
      writer.write(neff[m]);
    for (size_t m = neff.size()+1; m < max_M; m++)
      writer.write("");
    writer.newline();

/*
    //------------------------------------------------------------
    // test output

    std::cout << "************************************************************\n"
              << "milliseconds: " << time << std::endl
              << "************************************************************\n";
    size_t num_params = chains.num_params();
    for (size_t i = 0; i < num_params; i++) {
      std::cout << "------------------------------------------------------------\n";
      std::cout << "beta[" << i << "]" << std::endl;
      std::cout << "\tmean:        " << chains.mean(i) << std::endl;
      std::cout << "\tsd:          " << chains.sd(i) << std::endl;
      std::cout << "\tneff:        " << chains.effective_sample_size(i) << std::endl;
      std::cout << "\tsplit R hat: " << chains.split_potential_scale_reduction(i) << std::endl;
      }*/
    SUCCEED();
  }

  void test_logistic_speed_jags(const std::string& filename, 
        const size_t iterations,
                                const TestInfo& info) {
    if (!has_R)
      return;
    using std::vector;
    using boost::math::students_t;
    using boost::math::binomial;
    using boost::math::quantile;
    using std::setw;
    
    // 1) Get the generated beta.
    vector<double> beta;
    get_beta(filename, beta);
    
    // 2) Run JAGS num_chains times
    std::stringstream command;
    //command << path << get_path_separator() << "logistic"
    //<< " --data=" << path << get_path_separator() << filename << ".data.R"
    //<< " --iter=" << iterations
    //<< " --refresh=" << iterations;
    //vector<std::string> command_outputs;  
    //long time = run_stan(command.str(), filename, command_outputs);
    long time = 0;
    
    SUCCEED();
  }
};
const size_t LogisticSpeedTest::num_chains = 4;
bool LogisticSpeedTest::has_R;
bool LogisticSpeedTest::has_jags;
std::string LogisticSpeedTest::path;
std::ofstream LogisticSpeedTest::output_file;
std::string LogisticSpeedTest::Rscript;
std::vector<std::string> LogisticSpeedTest::data_files;
size_t LogisticSpeedTest::max_M;

TEST_F(LogisticSpeedTest,Prerequisites) {
  std::string command;
  command = "Rscript --version";
  try {
    run_command(command);
    has_R = true;
  } catch (...) {
    std::cout << "System does not have Rscript available" << std::endl
              << "Failed to run: " << command << std::endl;
  }

  std::vector<std::string> test_file;
  test_file.push_back("src");
  test_file.push_back("models");
  test_file.push_back("speed");
  test_file.push_back("empty.jags");
  command = "jags ";
  command += convert_model_path(test_file);
  
  try {
    run_command(command);
    has_jags = true;
  } catch (...) {
    std::cout << "System does not have jags available" << std::endl
              << "Failed to run: " << command << std::endl;
  }
}

TEST_F(LogisticSpeedTest,GenerateData) {
  if (!has_R) {
    std::cout << "No R available" << std::endl;
    return;  // should this fail?  probably
  }
  bool has_data = true;
  for (size_t i = 0; i < data_files.size() && has_data; i++) {
    std::string data_file = path;
    data_file += get_path_separator();
    data_file += data_files[i];
    data_file += ".data.R";
    std::ifstream file(data_file.c_str());
    if (!file)
      has_data = false;
  }

  if (has_data)
    return;

  // generate data using R script
  std::string command;
  command = "cd ";
  command += path;
  command += " && ";
  command += "Rscript ";
  command += Rscript;

  // no guarantee here that we have the right files

  ASSERT_NO_THROW(run_command(command))
    << command;
  SUCCEED();
}

TEST_P(LogisticSpeedTest, Stan) {
  std::stringstream filename;
  TestInfo info = GetParam();
  filename << "logistic_"
           << info.N
           << "_"
           << info.M;

  test_logistic_speed_stan(filename.str(), info.iterations, info);
}

TEST_P(LogisticSpeedTest, JAGS) {
  std::stringstream filename;
  TestInfo info = GetParam();
  filename << "logistic_"
           << info.N
           << "_"
           << info.M;
  
  test_logistic_speed_jags(filename.str(), info.iterations, info);
}


INSTANTIATE_TEST_CASE_P(,
                        LogisticSpeedTest,
                        testing::ValuesIn(getTestCases()));
