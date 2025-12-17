#!/bin/bash
# Master Script for Parallel Grid Search
# Runs all 3 parts simultaneously
# Usage: bash scripts/run_parallel.sh

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║          Starting Parallel Grid Search (3 processes)              ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

# Make scripts executable
chmod +x scripts/run_methods_part*.sh

# Start all 3 parts in background
echo "Starting Part 1..."
bash scripts/run_methods_part1.sh > logs/part1.log 2>&1 &
PID1=$!

echo "Starting Part 2..."
bash scripts/run_methods_part2.sh > logs/part2.log 2>&1 &
PID2=$!

echo "Starting Part 3..."
bash scripts/run_methods_part3.sh > logs/part3.log 2>&1 &
PID3=$!

echo ""
echo "All 3 processes started:"
echo "  Part 1 (PID: $PID1)"
echo "  Part 2 (PID: $PID2)"
echo "  Part 3 (PID: $PID3)"
echo ""
echo "Monitor logs:"
echo "  tail -f logs/part1.log"
echo "  tail -f logs/part2.log"
echo "  tail -f logs/part3.log"
echo ""

# Wait for all processes
wait $PID1 $PID2 $PID3

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║ ✓ All parallel grid search jobs completed!                        ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
