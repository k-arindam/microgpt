#!/bin/bash
# MicroGPT All-Languages Performance Benchmark
# Compiles (with maximum optimizations) and runs all 8 language implementations,
# then prints a ranked comparison table.
#
# Usage: bash benchmark.sh

set -e

BOLD="\033[1m"
GREEN="\033[32m"
CYAN="\033[36m"
YELLOW="\033[33m"
RED="\033[31m"
DIM="\033[2m"
RESET="\033[0m"

echo -e "${BOLD}╔═══════════════════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}║     MicroGPT — All Languages Performance Benchmark        ║${RESET}"
echo -e "${BOLD}║     Compile: max optimizations | Prefer AOT               ║${RESET}"
echo -e "${BOLD}╚═══════════════════════════════════════════════════════════╝${RESET}"
echo ""

# Ensure input.txt exists
if [ ! -f "input.txt" ]; then
    echo -e "${YELLOW}input.txt not found. Downloading...${RESET}"
    curl -sO https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt
    mv names.txt input.txt
fi

TIMEFILE=$(mktemp)
trap "rm -f $TIMEFILE" EXIT

# Storage arrays
LANG_NAMES=()
COMPILE_TIMES=()
RUN_TIMES=()
STATUSES=()

# ── Helpers ──────────────────────────────────────────
time_cmd() {
    # Runs a command, writes elapsed seconds to TIMEFILE
    local start=$(date +%s)
    "$@" > /dev/null 2>&1
    local end=$(date +%s)
    echo $((end - start)) > "$TIMEFILE"
}

check_tool() {
    command -v "$1" > /dev/null 2>&1
}

skip_lang() {
    local name="$1"
    local reason="$2"
    echo -e "  ${RED}✗ Skipped: ${reason}${RESET}"
    LANG_NAMES+=("$name")
    COMPILE_TIMES+=("-")
    RUN_TIMES+=("-")
    STATUSES+=("SKIP")
}

# ═══════════════════════════════════════════════════════
# Benchmark each language
# ═══════════════════════════════════════════════════════

# --- 1. Python ---
echo -e "${BOLD}[1/8] 🐍 Python${RESET}"
if check_tool python && [ -f "microgpt.py" ]; then
    echo -e "  ${DIM}No compilation needed (interpreted)${RESET}"
    echo -e "  ${CYAN}▶ Running: python microgpt.py${RESET}"
    time_cmd python microgpt.py
    PY_RUN=$(cat "$TIMEFILE")
    echo -e "  ${GREEN}✓ Done: ${PY_RUN}s${RESET}"
    LANG_NAMES+=("Python")
    COMPILE_TIMES+=("-")
    RUN_TIMES+=("$PY_RUN")
    STATUSES+=("OK")
else
    skip_lang "Python" "python not found or microgpt.py missing"
fi
echo ""

# --- 2. C++ (g++ -std=c++17 -O3) ---
echo -e "${BOLD}[2/8] ⚙️  C++${RESET}"
if check_tool g++ && [ -f "microgpt.cpp" ]; then
    echo -e "  ${CYAN}▶ Compiling: g++ -std=c++17 -O3${RESET}"
    time_cmd g++ -std=c++17 -O3 -o _bench_cpp microgpt.cpp
    CPP_COMPILE=$(cat "$TIMEFILE")
    echo -e "  ${GREEN}✓ Compiled: ${CPP_COMPILE}s${RESET}"
    echo -e "  ${CYAN}▶ Running: ./_bench_cpp${RESET}"
    time_cmd ./_bench_cpp
    CPP_RUN=$(cat "$TIMEFILE")
    echo -e "  ${GREEN}✓ Done: ${CPP_RUN}s${RESET}"
    rm -f _bench_cpp
    LANG_NAMES+=("C++ (g++ -O3)")
    COMPILE_TIMES+=("$CPP_COMPILE")
    RUN_TIMES+=("$CPP_RUN")
    STATUSES+=("OK")
else
    skip_lang "C++ (g++ -O3)" "g++ not found or microgpt.cpp missing"
fi
echo ""

# --- 3. Rust (rustc -O) ---
echo -e "${BOLD}[3/8] 🦀 Rust${RESET}"
if check_tool rustc && [ -f "microgpt.rs" ]; then
    echo -e "  ${CYAN}▶ Compiling: rustc -O${RESET}"
    time_cmd rustc -O microgpt.rs -o _bench_rs
    RS_COMPILE=$(cat "$TIMEFILE")
    echo -e "  ${GREEN}✓ Compiled: ${RS_COMPILE}s${RESET}"
    echo -e "  ${CYAN}▶ Running: ./_bench_rs${RESET}"
    time_cmd ./_bench_rs
    RS_RUN=$(cat "$TIMEFILE")
    echo -e "  ${GREEN}✓ Done: ${RS_RUN}s${RESET}"
    rm -f _bench_rs
    LANG_NAMES+=("Rust (rustc -O)")
    COMPILE_TIMES+=("$RS_COMPILE")
    RUN_TIMES+=("$RS_RUN")
    STATUSES+=("OK")
else
    skip_lang "Rust (rustc -O)" "rustc not found or microgpt.rs missing"
fi
echo ""

# --- 4. Swift (swiftc -O) ---
echo -e "${BOLD}[4/8] 🍎 Swift${RESET}"
if check_tool swiftc && [ -f "microgpt.swift" ]; then
    echo -e "  ${CYAN}▶ Compiling: swiftc -O${RESET}"
    time_cmd swiftc -O microgpt.swift -o _bench_swift
    SW_COMPILE=$(cat "$TIMEFILE")
    echo -e "  ${GREEN}✓ Compiled: ${SW_COMPILE}s${RESET}"
    echo -e "  ${CYAN}▶ Running: ./_bench_swift${RESET}"
    time_cmd ./_bench_swift
    SW_RUN=$(cat "$TIMEFILE")
    echo -e "  ${GREEN}✓ Done: ${SW_RUN}s${RESET}"
    rm -f _bench_swift
    LANG_NAMES+=("Swift (swiftc -O)")
    COMPILE_TIMES+=("$SW_COMPILE")
    RUN_TIMES+=("$SW_RUN")
    STATUSES+=("OK")
else
    skip_lang "Swift (swiftc -O)" "swiftc not found or microgpt.swift missing"
fi
echo ""

# --- 5. Dart (AOT: dart compile exe) ---
echo -e "${BOLD}[5/8] 🎯 Dart (AOT)${RESET}"
if check_tool dart && [ -f "microgpt.dart" ]; then
    echo -e "  ${CYAN}▶ Compiling: dart compile exe${RESET}"
    time_cmd dart compile exe microgpt.dart -o _bench_dart
    DART_COMPILE=$(cat "$TIMEFILE")
    echo -e "  ${GREEN}✓ Compiled: ${DART_COMPILE}s${RESET}"
    echo -e "  ${CYAN}▶ Running: ./_bench_dart${RESET}"
    time_cmd ./_bench_dart
    DART_RUN=$(cat "$TIMEFILE")
    echo -e "  ${GREEN}✓ Done: ${DART_RUN}s${RESET}"
    rm -f _bench_dart
    LANG_NAMES+=("Dart (AOT)")
    COMPILE_TIMES+=("$DART_COMPILE")
    RUN_TIMES+=("$DART_RUN")
    STATUSES+=("OK")
else
    skip_lang "Dart (AOT)" "dart not found or microgpt.dart missing"
fi
echo ""

# --- 6. Kotlin (kotlinc → java -jar) ---
echo -e "${BOLD}[6/8] 🟣 Kotlin${RESET}"
if check_tool kotlinc && check_tool java && [ -f "microgpt.kt" ]; then
    echo -e "  ${CYAN}▶ Compiling: kotlinc -include-runtime${RESET}"
    time_cmd kotlinc microgpt.kt -include-runtime -d _bench_kt.jar
    KT_COMPILE=$(cat "$TIMEFILE")
    echo -e "  ${GREEN}✓ Compiled: ${KT_COMPILE}s${RESET}"
    echo -e "  ${CYAN}▶ Running: java -jar _bench_kt.jar${RESET}"
    time_cmd java -jar _bench_kt.jar
    KT_RUN=$(cat "$TIMEFILE")
    echo -e "  ${GREEN}✓ Done: ${KT_RUN}s${RESET}"
    rm -f _bench_kt.jar
    LANG_NAMES+=("Kotlin (JVM)")
    COMPILE_TIMES+=("$KT_COMPILE")
    RUN_TIMES+=("$KT_RUN")
    STATUSES+=("OK")
else
    skip_lang "Kotlin (JVM)" "kotlinc/java not found or microgpt.kt missing"
fi
echo ""

# --- 7. JavaScript (Node.js) ---
echo -e "${BOLD}[7/8] 🟡 JavaScript${RESET}"
if check_tool node && [ -f "microgpt.js" ]; then
    echo -e "  ${DIM}No compilation needed (V8 JIT)${RESET}"
    echo -e "  ${CYAN}▶ Running: node microgpt.js${RESET}"
    time_cmd node microgpt.js
    JS_RUN=$(cat "$TIMEFILE")
    echo -e "  ${GREEN}✓ Done: ${JS_RUN}s${RESET}"
    LANG_NAMES+=("JavaScript (Node)")
    COMPILE_TIMES+=("-")
    RUN_TIMES+=("$JS_RUN")
    STATUSES+=("OK")
else
    skip_lang "JavaScript (Node)" "node not found or microgpt.js missing"
fi
echo ""

# --- 8. Go (go build -o) ---
echo -e "${BOLD}[8/8] 🔵 Go${RESET}"
if check_tool go && [ -f "microgpt.go" ]; then
    echo -e "  ${CYAN}▶ Compiling: go build${RESET}"
    time_cmd go build -o _bench_go microgpt.go
    GO_COMPILE=$(cat "$TIMEFILE")
    echo -e "  ${GREEN}✓ Compiled: ${GO_COMPILE}s${RESET}"
    echo -e "  ${CYAN}▶ Running: ./_bench_go${RESET}"
    time_cmd ./_bench_go
    GO_RUN=$(cat "$TIMEFILE")
    echo -e "  ${GREEN}✓ Done: ${GO_RUN}s${RESET}"
    rm -f _bench_go
    LANG_NAMES+=("Go (go build)")
    COMPILE_TIMES+=("$GO_COMPILE")
    RUN_TIMES+=("$GO_RUN")
    STATUSES+=("OK")
else
    skip_lang "Go (go build)" "go not found or microgpt.go missing"
fi

# ═══════════════════════════════════════════════════════
# Results Table — sorted by runtime
# ═══════════════════════════════════════════════════════

echo ""
echo ""
echo -e "${BOLD}╔═══════════════════════════════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}║                        BENCHMARK RESULTS                              ║${RESET}"
echo -e "${BOLD}╠═══════════════════════════════════════════════════════════════════════╣${RESET}"
echo -e "${BOLD}║                                                                       ║${RESET}"

# Build sortable entries: "runtime|index"
SORTED=""
for i in "${!LANG_NAMES[@]}"; do
    if [[ "${STATUSES[$i]}" == "OK" ]]; then
        SORTED+="${RUN_TIMES[$i]}|$i\n"
    fi
done

# Print header
echo -e "${BOLD}║  Rank │ Language              │ Compile (s) │  Run (s)  │ vs Fastest  ║${RESET}"
echo -e "${BOLD}║  ─────┼───────────────────────┼─────────────┼───────────┼─────────────║${RESET}"

RANK=1
FASTEST=""
while IFS='|' read -r rtime idx; do
    [[ -z "$rtime" ]] && continue
    name="${LANG_NAMES[$idx]}"
    ctime="${COMPILE_TIMES[$idx]}"

    if [[ -z "$FASTEST" ]]; then
        FASTEST="$rtime"
        ratio="1.00x"
    else
        ratio=$(awk "BEGIN {printf \"%.2fx\", $rtime / ($FASTEST > 0 ? $FASTEST : 1)}")
    fi

    # Pad fields for alignment
    printf "║  %4s │ %-21s │ %11s │ %7ss  │ %10s  ║\n" \
        "#${RANK}" "$name" "${ctime}s" "$rtime" "$ratio"
    RANK=$((RANK + 1))
done < <(echo -e "$SORTED" | sort -t'|' -k1 -n)

# Print skipped entries
for i in "${!LANG_NAMES[@]}"; do
    if [[ "${STATUSES[$i]}" == "SKIP" ]]; then
        printf "║  %4s │ %-21s │ %11s │ %9s │ %10s  ║\n" \
            "  -" "${LANG_NAMES[$i]}" "skipped" "skipped" "N/A"
    fi
done

echo -e "${BOLD}║                                                                       ║${RESET}"
echo -e "${BOLD}╚═══════════════════════════════════════════════════════════════════════╝${RESET}"
echo ""

# Cleanup any leftover binaries (safety)
rm -f _bench_cpp _bench_rs _bench_swift _bench_dart _bench_kt.jar _bench_go

echo -e "${GREEN}✓ Benchmark complete!${RESET}"
