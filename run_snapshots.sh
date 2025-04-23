#!/usr/bin/env bash

set -e

SNAPSHOTS_DIRS=(
  'Features/Shared/Foundation/DesignSystem/Tests/SnapshotTests/__Snapshots__'
  'Features/Shared/Product/DesignSystem/Select/Tests/SnapshotTests/__Snapshots__'
  'Apps/Applicant/UITests/Main/Tests/DesignSystem/Select/__Snapshots__'
)

SIMULATORS=(
  'name=iPhone 11 Tests,OS=18.2'
)

COMMITS_FILE="Tools/DesignSystem/SnapshotsTests/commits.txt"
ROOT_BRANCH="snapshot-tests/implementation"
WORKING_BRANCH="snapshot-tests/working"
WORKING_DIR="Output/SnapshotTestsExperiments"
WORKING_OUTPUTS="${WORKING_DIR}/Report"
ARTIFACTS_OUTPUTS="Output/Artifacts"
RUNS="${WORKING_OUTPUTS}/Runs"
LOGS="${WORKING_OUTPUTS}/Logs"

REPORTS=(
  "${ARTIFACTS_OUTPUTS}/UITestsJUnitReport.xml"
  "${ARTIFACTS_OUTPUTS}/UITestsStatsReport.csv"
  "${ARTIFACTS_OUTPUTS}/UnitTestsJUnitReport.xml"
  "${ARTIFACTS_OUTPUTS}/UnitTestsStatsReport.csv"
)

PROBES_COUNT=20
VERBOSE=true
DRY_RUN=false
USE_XCODEBUILD=false

function usage() {
  echo "Usage: $0 <command>"
  echo "  commits - Collect probing commits"
  echo "  run_list - Collect snapshots for list of commits"
}

function verbose() {
  if [ "$VERBOSE" = "true" ]; then
    log "\033[0;90m // $1\033[0m"
  fi
}

function info() {
  log
  log "\033[0;32m$1\033[0m"
  log "\033[0;32m---\033[0m"
}

function log() {
  echo -e $1
}

function fatal_error() {
  log "\033[0;31m$1\033[0m"
  exit 1
}

function collect_probing_commits() {
    local from_commit="snapshot-tests/begin"
    local to_commit="snapshot-tests/end"

    while [[ $# -gt 0 ]]; do
      case "$1" in
        --from)
          from_commit="$2"
          shift 2
          ;;
        --to)
          to_commit="$2"
          shift 2
          ;;
        *)
          fatal_error "Unknown option: $1"
          exit 1
          ;;
        esac
    done

    local commits_count=`git rev-list --format="%B" --no-commit-header $from_commit..$to_commit | grep "into develop" | wc -l`
    local step=$(( commits_count / PROBES_COUNT ))
    if (( step < 1 )); then step=1; fi

    verbose "Collecting probing commits"
    verbose "  - Available commits count: ${commits_count}"
    verbose "  - Limit: ${PROBES_COUNT}"
    verbose "  - Step: ${step}"

    git rev-list --format="%h::%ci::%B" --reverse --no-commit-header $from_commit..$to_commit \
      | grep "into develop" \
      | awk -F '::' -v step=$step 'NR % step == 0 { print $1 }' \
      > "${COMMITS_FILE}"
}

function collect_snapshots() {
  if [ ! -f "$COMMITS_FILE" ]; then
    fatal_error "Commits file not found: $COMMITS_FILE"
  fi

  verbose "Prepearing working directory"
  rm -rf "${WORKING_DIR}"
  mkdir -p "${WORKING_OUTPUTS}"
  mkdir -p "${RUNS}"
  mkdir -p "${LOGS}"
  verbose "Working directory: ${WORKING_DIR}"
  cp "${COMMITS_FILE}" "${WORKING_OUTPUTS}/commits.txt"

  collect_device_information
  prepare_for_runs

  local index=0
  local prev_commit=`git rev-parse --short HEAD`
  while IFS= read -r commit; do
    ((index++))
    local run_name=`printf "%02d.%s" "$index" "$commit"`
    run_snapshot "$run_name" "$commit" "$prev_commit"
    prev_commit=$commit
  done < "${WORKING_OUTPUTS}/commits.txt"

  collect_report
}

function collect_device_information() {
  info "Collecting system information"

  verbose "Run date"
  date > "${WORKING_OUTPUTS}/date.txt"

  verbose "Run scutil --get ComputerName"
  scutil --get ComputerName > "${WORKING_OUTPUTS}/computer_name.txt"

  verbose "Run system_profiler SPHardwareDataType"
  system_profiler SPHardwareDataType > "${WORKING_OUTPUTS}/hardware.txt"

  verbose "Run sw_vers"
  sw_vers > "${WORKING_OUTPUTS}/sw_vers.txt"
}

function run_upsh() {
  local log_output=$2

  if [ "$DRY_RUN" = "true" ]; then
    verbose "Dry run. Skipping tests"
    return 0
  fi

  verbose "Run up.sh"
  ./Tools/up.sh > "${log_output}.up_sh.log"
}

function run_tests() {
  local simulator=$1
  local log_output=$2

  if [ "$DRY_RUN" = "true" ]; then
    verbose "Dry run. Skipping tests"
    return 0
  fi

  if [ "$USE_XCODEBUILD" = "true" ]; then
    verbose "Run tests for simulator $simulator"
    xcodebuild test \
      -workspace HH.xcworkspace \
      -scheme ApplicantHH \
      -testPlan ApplicantHH-DesignSystem \
      -destination "platform=iOS Simulator,$simulator" > "${log_output}.tests.log" || true
    return 0
  fi

  ./Tools/CI/ci-run.sh fastlane run_collect_snapshots \
    scheme:"ApplicantHH" || true
}

function prepare_for_runs() {
  info "Preparing for runs"
  verbose "Switch to root branch $ROOT_BRANCH"
  git checkout $ROOT_BRANCH

  verbose "Create working branch $WORKING_BRANCH"
  git branch -D $WORKING_BRANCH || true
  git checkout -b $WORKING_BRANCH

  for snapshots_dir in "${SNAPSHOTS_DIRS[@]}"; do
    rm -rf "${snapshots_dir}"
  done

  verbose "Run initial tests"
  run_upsh "$WORKING_OUTPUTS/initial"
  run_tests "${SIMULATORS[0]}" "$WORKING_OUTPUTS/initial"

  verbose "Commit initial state"
  git add .
  git commit -m "Initial state" || true
}

function run_snapshot() {
  local run_name=$1
  local commit=$2
  local prev_commit=$3
  local run_dir="${RUNS}/${run_name}"

  info "Running snapshot for $run_name"

  verbose "Snapshot working directory ${run_dir}"
  mkdir -p "${run_dir}"

  verbose "Collecting changes in DS from $prev_commit to $commit"
  git diff "$prev_commit..$commit" ./Features/Shared/Foundation/DesignSystem ./Features/Shared/Product/DesignSystem > "${run_dir}/ds_diff.diff"

  verbose "Collecting diff stats"
  git diff "$prev_commit..$commit" --stat ./Features/Shared/Foundation/DesignSystem ./Features/Shared/Product/DesignSystem > "${run_dir}/ds_diff_stats.txt"

  verbose "Merge next commit $commit"
  git merge --no-edit $commit

  run_upsh "$run_dir/output"

  for simulator in "${SIMULATORS[@]}"; do
    info "Simulator $simulator"
    verbose "Remove snapshots"
    for snapshots_dir in "${SNAPSHOTS_DIRS[@]}"; do
      rm -rf "${snapshots_dir}"
    done

    run_tests "$simulator" "$run_dir/output"

    verbose "Commit updates"
    git add .
    git commit -m "Snapshot for $commit. $simulator" || true

    verbose "Collecting snapshots"
    mkdir -p "${run_dir}/Snapshots/${simulator}"
    for snapshots_dir in "${SNAPSHOTS_DIRS[@]}"; do
      verbose "Copying ${snapshots_dir}"
      if [ -d "${snapshots_dir}" ]; then
        cp -r "${snapshots_dir}" "${run_dir}/Snapshots/${simulator}"
      fi
    done

    verbose "Collecting test reports"
    mkdir -p "${run_dir}/Snapshots/${simulator}/TestReports"
    for report in "${REPORTS[@]}"; do
      if [ -f "${report}" ]; then
        verbose "Copy ${report}"
        cp -r "${report}" "${run_dir}/Snapshots/${simulator}/TestReports"
      fi
    done
  done
}

function collect_report() {
  info "Collecting report"
  zip -r "Report.zip" "$WORKING_OUTPUTS"/*
  mkdir -p "$ARTIFACTS_OUTPUTS"
  mv "Report.zip" "$ARTIFACTS_OUTPUTS/Report.zip"
}

function components_versions() {
  ./Tools/DesignSystem/components.sh | jq '.components |= sort_by(.name)'
}

function main() {
  if [ "$#" -lt 1 ]; then
    usage
    exit 1
  fi

  case $1 in
    commits)
      collect_probing_commits "${@:2}"
      ;;
    run_list)
      collect_snapshots "${@:2}"
      ;;
    *)
      fatal_error "Unknown command: $1"
      ;;
  esac
}

main "$@"
