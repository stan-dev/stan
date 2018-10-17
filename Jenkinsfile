@Library('StanUtils')
import org.stan.Utils

def utils = new org.stan.Utils()

def setupCXX(failOnError = true) {
    errorStr = failOnError ? "-Werror " : ""
    writeFile(file: "make/local", text: "CXX=${env.CXX} ${errorStr}")
}

def setup(String pr) {
    script = """
        make math-revert
        make clean-all
        git clean -xffd
    """
    if (pr != '')  {
        prNumber = pr.tokenize('-').last()
        script += """
            cd lib/stan_math
            git fetch https://github.com/stan-dev/math +refs/pull/${prNumber}/merge:refs/remotes/origin/pr/${prNumber}/merge
            git checkout refs/remotes/origin/pr/${prNumber}/merge
        """
    }
    return script
}

def runTests(String testPath, Boolean separateMakeStep=true) {
    if (separateMakeStep) {
        sh "./runTests.py -j${env.PARALLEL} ${testPath} --make-only"
    }
    try { sh "./runTests.py -j${env.PARALLEL} ${testPath}" }
    finally { junit 'test/**/*.xml' }
}

def runTestsWin(String testPath) {
    bat "runTests.py -j${env.PARALLEL} ${testPath} --make-only"
    try { bat "runTests.py -j${env.PARALLEL} ${testPath}" }
    finally { junit 'test/**/*.xml' }
}

def deleteDirWin() {
    bat "attrib -r -s /s /d"
    deleteDir()
}

String cmdstan_pr() { params.cmdstan_pr ?: "downstream_tests" }

pipeline {
    agent none
    parameters {
        string(defaultValue: '', name: 'math_pr', description: "Leave blank "
                + "unless testing against a specific math repo pull request, "
                + "e.g. PR-640.")
        string(defaultValue: 'downstream_tests', name: 'cmdstan_pr',
          description: 'PR to test CmdStan upstream against e.g. PR-630')
    }
    options {
        skipDefaultCheckout()
        preserveStashes(buildCount: 7)
    }
    stages {
        stage('Kill previous builds') {
            when {
                not { branch 'develop' }
                not { branch 'master' }
                not { branch 'downstream_tests' }
            }
            steps {
                script {
                    utils.killOldBuilds()
                }
            }
        }
        stage("Make manuals") {
            agent { label 'docker' }
            steps {
                script {
                    checkout scm
                    def docImage = docker.build("seantalts/bookdown",
                                                "src/docs")
                    docImage.inside {
                        sh "make doc"
                        archiveArtifacts 'doc/*'
                    }
                    docImage.push()
                }
            }
        }
        stage('Linting & Doc checks') {
            agent any
            steps {
                script {
                    retry(3) { checkout scm }
                    sh setup(params.math_pr)
                    stash 'StanSetup'
                    setupCXX()
                    parallel(
                        CppLint: { sh "make cpplint" },
                        API_docs: { sh 'make doxygen' },
                    )
                    def docImage = docker.build("seantalts/bookdown",
                                                ".circleci/doc-docker/Dockerfile")
                    docImage.inside {
                        sh "make doc"
                        archiveArtifacts 'doc/*'
                    }
                    docImage.push()
                }
            }
            post {
                always {
                    warnings consoleParsers: [[parserName: 'CppLint']], canRunOnFailed: true
                    deleteDir()
                }
            }
        }
        stage('Unit tests') {
            parallel {
                stage('Windows Headers & Unit') {
                    agent { label 'windows' }
                    steps {
                        deleteDirWin()
                            unstash 'StanSetup'
                            setupCXX()
                            bat "make -j${env.PARALLEL} test-headers"
                            setupCXX(false)
                            runTestsWin("src/test/unit")
                    }
                    post { always { deleteDirWin() } }
                }
                stage('Unit') {
                    agent any
                    steps {
                        unstash 'StanSetup'
                        setupCXX(false)
                        runTests("src/test/unit")
                    }
                    post { always { deleteDir() } }
                }
            }
        }
        stage('Integration') {
            agent any
            steps {
                unstash 'StanSetup'
                setupCXX()
                runTests("src/test/integration", separateMakeStep=false)
            }
            post { always { deleteDir() } }
        }
        stage('Upstream CmdStan tests') {
            when { expression { env.BRANCH_NAME ==~ /PR-\d+/ ||
                                env.BRANCH_NAME == "downstream_tests" } }
            steps {
                build(job: "CmdStan/${cmdstan_pr()}",
                      parameters: [string(name: 'stan_pr',
                                          value: env.BRANCH_NAME == "downstream_tests" ? '' : env.BRANCH_NAME),
                                   string(name: 'math_pr', value: params.math_pr)])
            }
        }
        stage('Performance') {
            agent { label 'master' }
            steps {
                unstash 'StanSetup'
                setupCXX()
                sh """
                    ./runTests.py -j${env.PARALLEL} src/test/performance
                    cd test/performance
                    RScript ../../src/test/performance/plot_performance.R
                """
            }
            post {
                always {
                    retry(2) {
                        junit 'test/**/*.xml'
                        archiveArtifacts 'test/performance/performance.csv,test/performance/performance.png'
                        perfReport compareBuildPrevious: true, errorFailedThreshold: 0, errorUnstableThreshold: 0, failBuildIfNoResultFile: false, modePerformancePerTestCase: true, sourceDataFiles: 'test/performance/**.xml'
                    }
                    deleteDir()
                }
            }
        }
    }
    post {
        always {
            node("osx || linux") {
                warnings consoleParsers: [[parserName: 'GNU C Compiler 4 (gcc)']], canRunOnFailed: true
                warnings consoleParsers: [[parserName: 'Clang (LLVM based)']], canRunOnFailed: true
            }
        }
        success {
            script {
                utils.updateUpstream(env,'cmdstan')
                utils.mailBuildResults("SUCCESSFUL")
            }
        }
        unstable { script { utils.mailBuildResults("UNSTABLE", "stan-buildbot@googlegroups.com") } }
        failure { script { utils.mailBuildResults("FAILURE", "stan-buildbot@googlegroups.com") } }
    }
}
