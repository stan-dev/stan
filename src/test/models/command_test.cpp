/** 
 * This test tries out different command line options for a 
 * generated model.
 *
 */
#include <gtest/gtest.h>
#include <stdexcept>
#include <boost/algorithm/string.hpp>

class ModelCommand : public ::testing::TestWithParam<int> {
private:
  static char path_separator;
public:
  static std::vector<std::string> expected_help_options;
  static std::string model_path;
  static std::string samples_option;

  static char get_path_separator() {
    if (path_separator == 0) {
      FILE *in;
      if(!(in = popen("make path_separator --no-print-directory", "r")))
        throw std::runtime_error("\"make path_separator\" has failed.");
      path_separator = fgetc(in);
      pclose(in);
    }
    return path_separator;
  }
  
  
  void static SetUpTestCase() {
    expected_help_options.push_back("help");
    expected_help_options.push_back("data");
    expected_help_options.push_back("init");
    expected_help_options.push_back("samples");
    expected_help_options.push_back("append_samples");
    expected_help_options.push_back("seed");
    expected_help_options.push_back("chain_id");
    expected_help_options.push_back("iter");
    expected_help_options.push_back("warmup");
    expected_help_options.push_back("thin");
    expected_help_options.push_back("refresh");
    expected_help_options.push_back("leapfrog_steps");
    expected_help_options.push_back("max_treedepth");
    expected_help_options.push_back("epsilon");
    expected_help_options.push_back("epsilon_pm");
    expected_help_options.push_back("unit_mass_matrix");
    expected_help_options.push_back("delta");
    expected_help_options.push_back("gamma");
    expected_help_options.push_back("test_grad");

    model_path.append("models");
    model_path.append(1, get_path_separator());
    model_path.append("command");

    samples_option = " --samples=";
    samples_option.append(model_path);
    samples_option.append(".csv");
  }

  /** 
   * Runs the command provided and returns the system output
   * as a string.
   * 
   * @param command A command that can be run from the shell
   * @return the system output of the command
   */  
  std::string run_command(const std::string& command) {
    FILE *in;
    if(!(in = popen(command.c_str(), "r"))) {
      std::string err_msg;
      err_msg = "Could not run: \"";
      err_msg+= command;
      err_msg+= "\"";
      throw std::runtime_error(err_msg.c_str());
    }

    std::string output;
    char buf[1024];
    size_t count = fread(&buf, 1, 1024, in);
    while (count > 0) {
      output += std::string(&buf[0], &buf[count]);
      count = fread(&buf, 1, 1024, in);
    }
    pclose(in);
    
    return output;
  }

  /** 
   * Returns the help options from the string provide.
   * Help options start with "--".
   * 
   * @param help_output output from "model/command --help"
   * @return a vector of strings of the help options
   */
  std::vector<std::string> get_help_options(const std::string& help_output) {    
    std::vector<std::string> help_options;

    size_t option_start = help_output.find("--");
    while (option_start != std::string::npos) {
      // find the option name (skip two characters for "--")
      option_start += 2;
      size_t option_end = help_output.find_first_of("= ", option_start);
      help_options.push_back(help_output.substr(option_start, option_end-option_start));
      option_start = help_output.find("--", option_start+1);
    }
        
    return help_options;
  }
  
  std::vector<std::pair<std::string, std::string> > 
  parse_output(const std::string& command_output) {
    std::vector<std::pair<std::string, std::string> > output;

    std::string option, value;
    size_t start = 0, end = command_output.find("\n", start);
    
    EXPECT_EQ("STAN SAMPLING COMMAND", 
              command_output.substr(start, end))
      << "command could not be run. output is: \n" 
      << command_output;
    if ("STAN SAMPLING COMMAND" != command_output.substr(start, end)) {
      return output;
    }
    start = end+1;
    end = command_output.find("\n", start);
    size_t equal_pos = command_output.find("=", start);

    while (equal_pos != std::string::npos) {
      using boost::trim;
      option = command_output.substr(start, equal_pos-start);
      value = command_output.substr(equal_pos+1, end - equal_pos - 1);
      trim(option);
      trim(value);
      output.push_back(std::pair<std::string, std::string>(option, value));
      start = end+1;
      end = command_output.find("\n", start);
      equal_pos = command_output.find("=", start);
    }
    return output;
  }
};

std::vector<std::string> ModelCommand::expected_help_options;
std::string ModelCommand::model_path;
std::string ModelCommand::samples_option;
char ModelCommand::path_separator = 0;

TEST_F(ModelCommand, HelpOptionsMatch) {
  std::string help_command = model_path;
  help_command.append(" --help");

  std::vector<std::string> help_options = 
    get_help_options(run_command(help_command));

  ASSERT_EQ(expected_help_options.size(), help_options.size());
  for (size_t i = 0; i < expected_help_options.size(); i++) {
    EXPECT_EQ(expected_help_options[i], help_options[i]);
  }
}

TEST_F(ModelCommand, DataOption) {
  std::string data_file_base;
  data_file_base.append("src");
  data_file_base.append(1, get_path_separator());
  data_file_base.append("test");
  data_file_base.append(1, get_path_separator());
  data_file_base.append(model_path);

  std::string data_command1 = model_path;
  data_command1.append(" --data=");
  data_command1.append(data_file_base);
  data_command1.append("1.Rdata");
  data_command1.append(samples_option);

  std::vector<std::pair<std::string, std::string> > output = 
    parse_output(run_command(data_command1));
  ASSERT_EQ(expected_help_options.size()-2, output.size()) 
    << "Should have the same options without \"help\" and \"append_samples\"";
  for (size_t i = 0; i < output.size(); i++) {
    if (output[i].first == "data") {
      std::string csv_file_name = data_file_base;
      csv_file_name.append("1.Rdata");
      EXPECT_EQ(csv_file_name, output[i].second);
    }
  }


  std::string data_command2 = model_path;
  data_command2.append(" --data=");
  data_command2.append(data_file_base);
  data_command2.append("2.Rdata");
  data_command2.append(samples_option);

  output = parse_output(run_command(data_command2));
  ASSERT_EQ(expected_help_options.size()-2, output.size()) 
    << "Should have the same options without \"help\" and \"append_samples\"";
  for (size_t i = 0; i < output.size(); i++) {
    if (output[i].first == "data") {
      std::string csv_file_name = data_file_base;
      csv_file_name.append("2.Rdata");
      EXPECT_EQ(csv_file_name, output[i].second);
    }
  }

  
  
}
/*TEST_P(ModelCommand, PTest) {
std::cout << "PTest: " << GetParam() << std::endl;

}*/

//INSTANTIATE_TEST_CASE_P(abc, ModelCommand, testing::Range(1, 10));

