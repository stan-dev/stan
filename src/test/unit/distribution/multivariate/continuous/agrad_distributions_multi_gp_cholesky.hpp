class agrad_distributions_multi_gp_cholesky : public ::testing::Test {
protected:
  virtual void SetUp() {
    y.resize(3,2);
    y << 2.0, -2.0, 11.0,
         0.0, 1.0, 5.0;
    y2.resize(3,2);
    y2 << 15.0, 1.0, -5.0,
          2.0, 3.0, -10.0;

    w.resize(3,1);
    w << 1.0, 1.0, 3.0;
    w2.resize(3,1);
    w2 << 6.0, 2.0, 6.0;

    Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> Sigma(2,2);
    Sigma << 9.0, -3.0,
            -3.0,  4.0;
    L = Sigma.llt().matrixL();

    Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> Sigma2(2,2);
    Sigma2 << 3.0, 1.0,
              1.0, 5.0;
    L2 = Sigma2.llt().matrixL();
  }
  
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> y;
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> y2;
  Eigen::Matrix<double,Eigen::Dynamic,1> w;
  Eigen::Matrix<double,Eigen::Dynamic,1> w2;
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> L;
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> L2;
};
