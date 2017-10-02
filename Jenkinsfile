def clean() {
    sh """
        make math-revert
        make clean-all
        git clean -xffd
        echo 'CC=${env.CXX}' > make/local
    """
}

def checkout_pr(String pr) {
    if (pr != '')  {
        prNumber = pr.tokenize('-').last()
        sh """ cd stan/lib/stan_math
            git fetch https://github.com/stan-dev/math +refs/pull/${prNumber}/merge:refs/remotes/origin/pr/${prNumber}/merge
            git checkout refs/remotes/origin/pr/${prNumber}/merge
        """
    }
}

pipeline {
    agent any
    options {
        disableConcurrentBuilds()
    }
    parameters {
        string(defaultValue: '', name: 'math_pr')
    }
    stages {
        stage('Clean & Setup') {
            steps {
                clean()
                checkout_pr(params.math_pr)
                sh "echo 'CXXFLAGS += -Werror' >> make/local"
            }
        }
        //stage('Linting & Doc checks') {
        //    steps {
        //        parallel(
        //            CppLint: { sh "make cpplint" },
        //            documentation: { sh 'make doxygen' },
        //            manual: { sh 'make manual' },
        //            headers: { sh "make -j${env.PARALLEL} test-headers" },
        //            failFast: true
        //        )
        //    }
        //}
        stage('Tests') {
            failFast true
            parallel {
                stage('Windows Unit') {
                    agent { label 'windows' }
                    clean()
                    checkout_pr(params.math_pr)
                    sh "./runTests.py -j${env.PARALLEL} src/test/unit"
                }
                stage('Windows Headers') { 
                    node { label 'windows' }
                    clean()
                    checkout_pr(params.math_pr)
                    sh "make -j${env.PARALLEL} test-headers"
                }
                //These aren't turned on in the old config - broken
                //windowsIntegration: {
                //    agent { label 'windows' }
                //    clean()
                //    sh "./runTests.py -j${env.PARALLEL} src/test/integration"
                //},
                stage('Unit') { 
                    agent any
                    clean()
                    checkout_pr(params.math_pr)
                    sh "./runTests.py -j${env.PARALLEL} src/test/unit"
                }
                stage('Integration') {
                    agent any
                    clean()
                    checkout_pr(params.math_pr)
                    sh "./runTests.py -j${env.PARALLEL} src/test/integration"
                }
            }
        }
        stage('Upstream CmdStan tests') {
            when {
                allOf {
                    not { branch 'master' }
                    not { branch 'develop' }
                }
            }
            steps {
                build(job: 'CmdStan Pipeline/develop',
                        parameters: [string(name: 'stan_pr', value: env.BRANCH_NAME),
                                        string(name: 'math_pr', value: params.math_pr)])
            }
        }
            
        stage('Performance') {
            agent { label 'gelman-group-mac' }
            //XXX Eventually make this block from running with other jobs
            //buildBlocker blockingJobs: '.*', blockLevel: 'NODE'
            steps {
                clean()
                checkout_pr(params.math_pr)
                sh """
                    ./runTests.py src/test/performance
                    cd test/performance
                    RScript ../../src/test/performance/plot_performance.R 
                """
            }
        }
        stage('Update CmdStan Upstream') {
            agent none
            when { branch "develop" }
            steps {
                sh "curl -O https://raw.githubusercontent.com/stan-dev/ci-scripts/master/jenkins/create-cmdstan-pull-request.sh"
                sh "sh create-cmdstan-pull-request.sh"
            }
        }
    }
    post {
        always {
            warnings consoleParsers: [[parserName: 'GNU C Compiler 4 (gcc)']], canRunOnFailed: true
            warnings consoleParsers: [[parserName: 'Clang (LLVM based)']], canRunOnFailed: true
            warnings consoleParsers: [[parserName: 'CppLint']], canRunOnFailed: true
            archiveArtifacts 'test/performance/performance.csv,test/performance/performance.png'
            junit 'test/**/*.xml'
            perfReport compareBuildPrevious: true, errorFailedThreshold: 0, errorUnstableThreshold: 0, failBuildIfNoResultFile: false, modePerformancePerTestCase: true, sourceDataFiles: 'test/performance/**.xml'
        }
    }
}
