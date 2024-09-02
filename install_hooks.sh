#!/bin/sh

# Get the .git directory
GIT_DIR=$(git rev-parse --git-dir)

# Create symbolic link for pre-commit hook
ln -sf "../../.hooks/pre-commit" "${GIT_DIR}/hooks/pre-commit"

echo "Git hooks installed successfully."
echo "Please ensure you have pre-commit installed: pip install pre-commit"
echo "Then run: pre-commit install && pre-commit install --hook-type commit-msg"
