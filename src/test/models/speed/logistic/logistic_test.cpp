#include <gtest/gtest.h>
#include <test/models/utility.hpp>
#include <fstream>

const size_t num_chains = 4;
bool has_R = false;
bool has_jags = false;
std::vector<std::string> model_path;
std::string Rscript;
std::vector<std::string> data_files;

TEST(LogisticSpeedTest,Prerequisites) {
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

  model_path.push_back("models");
  model_path.push_back("speed");
  model_path.push_back("logistic");
  Rscript = "logistic_generate_data.R";

  data_files.push_back("logistic_1000_1");
  data_files.push_back("logistic_1000_10");
  data_files.push_back("logistic_1000_100");
  data_files.push_back("logistic_1000_500");
  data_files.push_back("logistic_5000_1");
  data_files.push_back("logistic_5000_10");
  data_files.push_back("logistic_5000_100");
  data_files.push_back("logistic_5000_500");
  data_files.push_back("logistic_5000_1000");
  data_files.push_back("logistic_10000_1");
  data_files.push_back("logistic_10000_10");
  data_files.push_back("logistic_10000_100");
  data_files.push_back("logistic_10000_500");
  data_files.push_back("logistic_10000_1000");
}

TEST(LogisticSpeedTest,GenerateData) {
  if (!has_R) {
    std::cout << "No R available" << std::endl;
    return;
  }
  bool has_data = true;
  std::string path = convert_model_path(model_path);
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
  command += convert_model_path(model_path);
  command += " && ";
  command += "Rscript ";
  command += Rscript;

  ASSERT_NO_THROW(run_command(command))
    << command;
  SUCCEED();
}

TEST(LogisticSpeedTest,Stan_1000_1) {
  std::string filename = "logistic_1000_1";
  std::stringstream command;
  std::string path = convert_model_path(model_path);


  command << path << get_path_separator() << "logistic"
          << " --data=" << path << get_path_separator() << filename << ".Rdata"
          << " --iter=2000";
  
  
  for (size_t chain = 0; chain < num_chains; chain++) {
    std::stringstream command_chain;
    command_chain << command.str();
    command_chain << " --chain_id=" << chain
                  << " --samples=" << path << get_path_separator() 
                  << filename << ".chain_" << chain << ".csv";
    std::cout << command_chain.str() << std::endl;
  }
  
  
  SUCCEED();
}
