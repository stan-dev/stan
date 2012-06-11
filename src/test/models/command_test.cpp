/** 
 * This test tries out different command line options for a 
 * generated model.
 *
 */
#include <gtest/gtest.h>
#include <stdexcept>
#include <boost/algorithm/string.hpp>
#include <stan/mcmc/chains.hpp>

using std::vector;
using std::string;
using std::pair;

class ModelCommand : public ::testing::TestWithParam<int> {
private:
  static char path_separator;
public:  
  static vector<string> expected_help_options;
  static string model_path;
  static string samples_option;

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
  string run_command(const string& command) {
    FILE *in;
    if(!(in = popen(command.c_str(), "r"))) {
      string err_msg;
      err_msg = "Could not run: \"";
      err_msg+= command;
      err_msg+= "\"";
      throw std::runtime_error(err_msg.c_str());
    }

    string output;
    char buf[1024];
    size_t count = fread(&buf, 1, 1024, in);
    while (count > 0) {
      output += string(&buf[0], &buf[count]);
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
  vector<string> get_help_options(const string& help_output) {    
    vector<string> help_options;

    size_t option_start = help_output.find("--");
    while (option_start != string::npos) {
      // find the option name (skip two characters for "--")
      option_start += 2;
      size_t option_end = help_output.find_first_of("= ", option_start);
      help_options.push_back(help_output.substr(option_start, option_end-option_start));
      option_start = help_output.find("--", option_start+1);
    }
        
    return help_options;
  }
  
  vector<pair<string, string> > 
  parse_output(const string& command_output) {
    vector<pair<string, string> > output;

    string option, value;
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

    while (equal_pos != string::npos) {
      using boost::trim;
      option = command_output.substr(start, equal_pos-start);
      value = command_output.substr(equal_pos+1, end - equal_pos - 1);
      trim(option);
      trim(value);
      output.push_back(pair<string, string>(option, value));
      start = end+1;
      end = command_output.find("\n", start);
      equal_pos = command_output.find("=", start);
    }
    return output;
  }

  void check_output(const string& command_output,
                    const vector<pair<string, string> >& defaults) {
    vector<pair<string, string> > expected_output;
    expected_output.push_back(pair<string,string>("data", 
                                                  "(specified model requires no data)"));
    expected_output.push_back(pair<string,string>("init", 
                                                  "random initialization"));
    expected_output.push_back(pair<string,string>("init tries", 
                                                  "1"));
    expected_output.push_back(pair<string,string>("samples", 
                                                  model_path+".csv"));    
    expected_output.push_back(pair<string,string>("append_samples",
                                                  "0"));    
    expected_output.push_back(pair<string,string>("seed", 
                                                  ""));    
    expected_output.push_back(pair<string,string>("chain_id", 
                                                  "1 (default)"));    
    expected_output.push_back(pair<string,string>("iter", 
                                                  "2000"));    
    expected_output.push_back(pair<string,string>("warmup", 
                                                  "1000"));    
    expected_output.push_back(pair<string,string>("thin", 
                                                  "1 (default)"));    
    expected_output.push_back(pair<string,string>("unit_mass_matrix", 
                                                  "0"));    
    expected_output.push_back(pair<string,string>("leapfrog_steps", 
                                                  "-1"));    
    expected_output.push_back(pair<string,string>("max_treedepth", 
                                                  "10"));    
    expected_output.push_back(pair<string,string>("epsilon", 
                                                  "-1"));    
    expected_output.push_back(pair<string,string>("epsilon_pm", 
                                                  "0"));    
    expected_output.push_back(pair<string,string>("delta", 
                                                  "0.5"));
    expected_output.push_back(pair<string,string>("gamma", 
                                                  "0.05"));    

    for (size_t i = 0; i < defaults.size(); i++) {
      for (size_t j = 0; j < expected_output.size(); j++) {
        if (expected_output[j].first == defaults[i].first) {
          expected_output[j].second = defaults[i].second;
          break;
        }
      }
    }
    
    vector<pair<string, string> > output = parse_output(command_output);
    ASSERT_EQ(expected_output.size(), output.size());
    for (size_t i = 0; i < expected_output.size(); i++) {
      EXPECT_EQ(expected_output[i].first, output[i].first) <<
        "Order of output should match";
      if (expected_output[i].first == "seed" && expected_output[i].second == "") {
        // when seed is default, check to see that it is randomly generated
        if (boost::algorithm::ends_with(output[i].second, "(randomly generated)"))
          SUCCEED();
        else
          ADD_FAILURE() <<
            output[i].first << " is not randomly generated: " << output[i].second;
      } else {
        EXPECT_EQ(expected_output[i].second, output[i].second)
          << "Option " << expected_output[i].first << " returned unexpected value";
      }
        
    }

  }
  void check_output(const string& command_output) {
    vector<pair<string, string> > defaults;
    check_output(command_output, defaults);
  }
};

vector<string> ModelCommand::expected_help_options;
string ModelCommand::model_path;
string ModelCommand::samples_option;
char ModelCommand::path_separator = 0;

TEST_F(ModelCommand, HelpOptionsMatch) {
  string help_command = model_path;
  help_command.append(" --help");

  vector<string> help_options = 
    get_help_options(run_command(help_command));

  ASSERT_EQ(expected_help_options.size(), help_options.size());
  for (size_t i = 0; i < expected_help_options.size(); i++) {
    EXPECT_EQ(expected_help_options[i], help_options[i]);
  }
}

TEST_F(ModelCommand, DataOption) {
  string data_file_base;
  data_file_base.append("src");
  data_file_base.append(1, get_path_separator());
  data_file_base.append("test");
  data_file_base.append(1, get_path_separator());
  data_file_base.append(model_path);

  string data_command1 = model_path;
  data_command1.append(" --data=");
  data_command1.append(data_file_base);
  data_command1.append("1.Rdata");
  data_command1.append(samples_option);

  vector<pair<string, string> > defaults;

  defaults.push_back(pair<string, string>("data", data_file_base+"1.Rdata"));
  check_output(run_command(data_command1), defaults);
  // test sampled values
  vector<string> names;
  vector<vector<size_t> > dimss;
  stan::mcmc::read_variables(model_path+".csv", 2U,
                             names, dimss);
      
  stan::mcmc::chains<> c1(1U, names, dimss);
  stan::mcmc::add_chain(c1, 0, model_path+".csv", 2U);
  EXPECT_NEAR(c1.mean(0U), 0, 1);


  string data_command2 = model_path;
  data_command2.append(" --data=");
  data_command2.append(data_file_base);
  data_command2.append("2.Rdata");
  data_command2.append(samples_option);

  
  defaults.clear();
  defaults.push_back(pair<string, string>("data", data_file_base+"2.Rdata"));
  check_output(run_command(data_command2), defaults);
  // test sampled values
  stan::mcmc::chains<> c2(1U, names, dimss);
  stan::mcmc::add_chain(c2, 0, model_path+".csv", 2U);
  EXPECT_NEAR(c2.mean(0U), 100, 1);

}
/*TEST_P(ModelCommand, PTest) {
std::cout << "PTest: " << GetParam() << std::endl;

}*/

//INSTANTIATE_TEST_CASE_P(abc, ModelCommand, testing::Range(1, 10));

