#include <gtest/gtest.h>
#include <stdexcept>

class Models_BugsExamples_Vol3_Seeds : public ::testing::Test {
protected:
  virtual void SetUp() {
    FILE *in;
    if(!(in = popen("make path_separator --no-print-directory", "r")))
      throw std::runtime_error("\"make path_separator\" has failed.");
    path_separator += fgetc(in);
    pclose(in);
    
    model.append("models").append(path_separator);
    model.append("bugs_examples").append(path_separator);
    model.append("vol3").append(path_separator);
    model.append("data_cloning").append(path_separator);
    model.append("seeds");

    output1 = model + "1.csv";
    output2 = model + "2.csv";

    data = model + ".Rdata";
  }
  std::string path_separator;
  std::string model;
  std::string output1;
  std::string output2;
  std::string data;
};

TEST_F(Models_BugsExamples_Vol3_Seeds,RunModel) {
  std::string command;
  command = model;
  command += " --samples=";
  command += output1;
  command += " --data=";
  command += data;
  EXPECT_EQ(0, system(command.c_str())) 
    << "Can not execute command: " << command << std::endl;
            
  
  command = model;
  command += " --samples=";
  command += output2;
  command += " --data=";
  command += data;
  EXPECT_EQ(0, system(command.c_str()))
    << "Can not execute command: " << command << std::endl;
}
