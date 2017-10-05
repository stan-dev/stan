def setup(String pr) {
    script = """
        make math-revert
        make clean-all
        git clean -xffd
        echo CC=${env.CXX} -Werror > make/local
    """
    if (pr != '')  {
        prNumber = pr.tokenize('-').last()
        script += """ 
            cd stan/lib/stan_math
            git fetch https://github.com/stan-dev/math +refs/pull/${prNumber}/merge:refs/remotes/origin/pr/${prNumber}/merge
            git checkout refs/remotes/origin/pr/${prNumber}/merge
        """
    }
    return script
}

pipeline {
    agent none
    options {
        disableConcurrentBuilds()
    }
    parameters {
        string(defaultValue: '', name: 'math_pr')
    }
    stages {
        stage('Linting & Doc checks') {
            agent any
            steps {
                script {
                    sh setup(params.math_pr)
                    parallel(
                        CppLint: {
                            sh "make cpplint"
                        },
                        documentation: {
                            sh 'make doxygen' 
                        },
                        manual: {
                            sh 'make manual' 
                        },
                        headers: {
                            sh "make -j${env.PARALLEL} test-headers" 
                        },
                        failFast: true
                    )
                }
            }
            post { 
                always {
                    cleanWs()
                    warnings consoleParsers: [[parserName: 'CppLint']], canRunOnFailed: true
                    warnings consoleParsers: [[parserName: 'GNU C Compiler 4 (gcc)']], canRunOnFailed: true
                    warnings consoleParsers: [[parserName: 'Clang (LLVM based)']], canRunOnFailed: true
                }
            }
        }
        stage('Tests') {
            failFast true
            parallel {
                stage('Windows Unit') {
                    agent { label 'windows' }
                    steps {
                        bat setup(params.math_pr)
                        bat "runTests.py -j${env.PARALLEL} src/test/unit"
                        junit 'test/**/*.xml'
                    }
                    post {
                        always { 
                            cleanWs() 
                            warnings consoleParsers: [[parserName: 'GNU C Compiler 4 (gcc)']], canRunOnFailed: true
                            warnings consoleParsers: [[parserName: 'Clang (LLVM based)']], canRunOnFailed: true
                        }
                    }
                }
                stage('Windows Headers') { 
                    agent { label 'windows' }
                    steps {
                        bat setup(params.math_pr)
                        bat "make -j${env.PARALLEL} test-headers"
                    }
                    post {
                        always {
                            cleanWs() 
                            warnings consoleParsers: [[parserName: 'GNU C Compiler 4 (gcc)']], canRunOnFailed: true
                            warnings consoleParsers: [[parserName: 'Clang (LLVM based)']], canRunOnFailed: true
                        }
                    }
                }
                stage('Unit') { 
                    agent any
                    steps {
                        sh setup(params.math_pr)
                        sh "./runTests.py -j${env.PARALLEL} src/test/unit"
                        junit 'test/**/*.xml'
                    }
                    post {
                        always {
                            cleanWs() 
                            warnings consoleParsers: [[parserName: 'GNU C Compiler 4 (gcc)']], canRunOnFailed: true
                            warnings consoleParsers: [[parserName: 'Clang (LLVM based)']], canRunOnFailed: true
                        }
                    }
                }
                stage('Integration') {
                    agent any
                    steps { 
                        sh setup(params.math_pr)
                        sh "./runTests.py -j${env.PARALLEL} src/test/integration"
                        junit 'test/**/*.xml'
                      }
                    post {
                        always {
                            cleanWs() 
                            warnings consoleParsers: [[parserName: 'GNU C Compiler 4 (gcc)']], canRunOnFailed: true
                            warnings consoleParsers: [[parserName: 'Clang (LLVM based)']], canRunOnFailed: true
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
                        build(job: 'CmdStan/downstream tests',
                                parameters: [string(name: 'stan_pr', value: env.BRANCH_NAME),
                                                string(name: 'math_pr', value: params.math_pr)])
                    }
                }
            }
        }
        stage('Performance') {
            agent { label 'gelman-group-mac' }
            steps {
                sh setup(params.math_pr)
                sh """
                    ./runTests.py -j${env.PARALLEL} src/test/performance
                    cd test/performance
                    RScript ../../src/test/performance/plot_performance.R 
                """
                junit 'test/**/*.xml'
                archiveArtifacts 'test/performance/performance.csv,test/performance/performance.png'
                perfReport compareBuildPrevious: true, errorFailedThreshold: 0, errorUnstableThreshold: 0, failBuildIfNoResultFile: false, modePerformancePerTestCase: true, sourceDataFiles: 'test/performance/**.xml'
                warnings consoleParsers: [[parserName: 'GNU C Compiler 4 (gcc)']], canRunOnFailed: true
                warnings consoleParsers: [[parserName: 'Clang (LLVM based)']], canRunOnFailed: true
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
}
