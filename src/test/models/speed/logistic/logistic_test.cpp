#include <gtest/gtest.h>
#include <test/models/utility.hpp>
#include <fstream>
#include <stan/io/dump.hpp>
#include <stan/mcmc/chains.hpp>
#include <boost/date_time/posix_time/posix_time_types.hpp>

class LogisticSpeedTest :
  public testing::Test {
public:
  static const size_t num_chains;
  static bool has_R;
  static bool has_jags;
  static std::string path;

  std::string Rscript;
  std::vector<std::string> data_files;

  static void SetUpTestCase() {
    std::vector<std::string> model_path;
    model_path.push_back("models");
    model_path.push_back("speed");
    model_path.push_back("logistic");

    path = convert_model_path(model_path);
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
    using boost::posix_time::ptime;

    long time = 0;
    for (size_t chain = 0; chain < num_chains; chain++) {
      std::stringstream command_chain;
      command_chain << command;
      command_chain << " --chain_id=" << chain
                    << " --samples=" << path << get_path_separator() 
                    << filename << ".chain_" << chain << ".csv";
      std::string command_output;
      try {
        ptime time_start(boost::posix_time::microsec_clock::universal_time()); // start timer
        command_output = run_command(command_chain.str());
        ptime time_end(boost::posix_time::microsec_clock::universal_time());   // end timer
        time += (time_end - time_start).total_milliseconds();
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
            << filename << ".chain_0.csv";
  
    std::vector<std::string> names;
    std::vector<std::vector<size_t> > dimss;
    stan::mcmc::read_variables(samples.str(), 2U,
                               names, dimss);

    stan::mcmc::chains<> chains(num_chains, names, dimss);
    for (size_t chain = 0; chain < num_chains; chain++) {
      samples.str("");
      samples << path << get_path_separator()
              << filename << ".chain_" << chain << ".csv";
      stan::mcmc::add_chain(chains, chain, samples.str(), 2U);
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
                   << "_param.Rdata";
    std::ifstream param_ifstream(param_filename.str().c_str());
    stan::io::dump param_values(param_ifstream);
  
    beta = param_values.vals_r("beta");
    for (size_t i = 0; i < beta.size(); i++) {
      std::cout << "beta[" << i << "]: " << beta[i] << std::endl;
    }
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
  void test_logistic_speed_stan(const std::string& filename, const size_t iterations) {
    if (!has_R)
      return;

    // 1) Get the generated beta.
    std::vector<double> beta;
    get_beta(filename, beta);

    // 2) Run Stan num_chains times
    std::stringstream command;
    command << path << get_path_separator() << "logistic"
            << " --data=" << path << get_path_separator() << filename << ".Rdata"
            << " --iter=" << iterations
            << " --refresh=" << iterations;
    std::vector<std::string> command_outputs;  
    long time = run_stan(command.str(), filename, command_outputs);


    // 3) Test values of sampled parameters
    stan::mcmc::chains<> chains = create_chains(filename);
    // TODO: test sampled values using chain


    // 4) Output useful values.


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
    }
    SUCCEED();
  }

};
const size_t LogisticSpeedTest::num_chains = 4;
bool LogisticSpeedTest::has_R;
bool LogisticSpeedTest::has_jags;
std::string LogisticSpeedTest::path;
  

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
  command += path;
  
  try {
    run_command(command);
    has_jags = true;
  } catch (...) {
    std::cout << "System does not have jags available" << std::endl
              << "Failed to run: " << command << std::endl;
  }

  Rscript = "logistic_generate_data.R";

  data_files.push_back("logistic_128_2");
  data_files.push_back("logistic_1024_2");
  data_files.push_back("logistic_4096_2");
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
    data_file += ".Rdata";
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


TEST_F(LogisticSpeedTest,Stan_128_2) { 
  test_logistic_speed_stan("logistic_128_2", 250U);
}

TEST_F(LogisticSpeedTest,Stan_1024_2) { 
  test_logistic_speed_stan("logistic_1024_2", 250U);
}

TEST_F(LogisticSpeedTest,Stan_4096_2) { 
  test_logistic_speed_stan("logistic_4096_2", 250U);
}


