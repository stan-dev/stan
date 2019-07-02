@Library('StanUtils')
import org.stan.Utils

def utils = new org.stan.Utils()

def setupCXX(failOnError = true) {
    errorStr = failOnError ? "-Werror " : ""
    writeFile(file: "make/local", text: "CXX=${env.CXX} ${errorStr}")
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
String stan_pr() {
    if (env.BRANCH_NAME == 'downstream_tests') {
        ''
    } else if (env.BRANCH_NAME == 'downstream_hotfix') {
        'master'
    } else {
        env.BRANCH_NAME
    }
}

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
        stage('Linting & Doc checks') {
            agent any
            steps {
                script {
                    sh "printenv"
                    retry(3) { checkout scm }
                    sh """
                       make math-revert
                       make clean-all
                       git clean -xffd
                    """
                    utils.checkout_pr("math", "lib/stan_math", params.math_pr)
                    stash 'StanSetup'
                    setupCXX()
                    parallel(
                        CppLint: { sh "make cpplint" },
                        API_docs: { sh 'make doxygen' },
                    )
                }
            }
            post {
                always {

                    recordIssues id: "lint_doc_checks", 
                    name: "Linting & Doc checks",
                    enabledForFailure: true, 
                    aggregatingResults : true, 
                    tools: [
                        cppLint(id: "cpplint", name: "Linting & Doc checks@CPPLINT")
                    ],
                    blameDisabled: false,
                    qualityGates: [[threshold: 1, type: 'TOTAL', unstable: true]],
                    healthy: 10, unhealthy: 100, minimumSeverity: 'HIGH',
                    referenceJobName: env.BRANCH_NAME

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
                                env.BRANCH_NAME == "downstream_tests" ||
                                env.BRANCH_NAME == "downstream_hotfix" } }
            steps {
                build(job: "CmdStan/${cmdstan_pr()}",
                      parameters: [string(name: 'stan_pr', value: stan_pr()),
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
                recordIssues id: "pipeline", 
                name: "Entire pipeline results",
                enabledForFailure: true, 
                aggregatingResults : false, 
                tools: [
                    gcc4(id: "pipeline_gcc4", name: "GNU C Compiler"),
                    clang(id: "pipeline_clang", name: "LLVM/Clang")
                ],
                blameDisabled: false,
                qualityGates: [[threshold: 30, type: 'TOTAL', unstable: true]],
                healthy: 10, unhealthy: 100, minimumSeverity: 'HIGH',
                referenceJobName: env.BRANCH_NAME
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
