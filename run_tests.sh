#!/bin/bash
# Script to run all tests

echo "Running Deep Learning NLP Test Suite"
echo "===================================="
echo ""

# Set PYTHONPATH to include src directory
export PYTHONPATH=src

# Run all tests
python -m pytest tests/ -v

echo ""
echo "Test run completed!"
