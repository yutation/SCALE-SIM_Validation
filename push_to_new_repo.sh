#!/bin/bash
# Script to push SCALE-Sim Validation to the new repository
# Repository: https://github.com/scalesim-project/SCALE-Sim-validation

set -e  # Exit on error

echo "=========================================="
echo "SCALE-Sim Validation - Repository Setup"
echo "=========================================="
echo ""

# Navigate to validation directory
cd /home/Owner/work/SCALE-Sim/validation

# Show current status
echo "Current repository status:"
git status
echo ""

# Add the new remote (if not already added)
echo "Adding new remote 'scalesim'..."
if git remote get-url scalesim 2>/dev/null; then
    echo "Remote 'scalesim' already exists. Updating URL..."
    git remote set-url scalesim https://github.com/scalesim-project/SCALE-Sim-validation.git
else
    git remote add scalesim https://github.com/scalesim-project/SCALE-Sim-validation.git
fi

echo ""
echo "Current remotes:"
git remote -v
echo ""

# Option to commit changes
printf "Do you want to commit the current changes? (y/n) "
read REPLY
if [ "$REPLY" = "y" ] || [ "$REPLY" = "Y" ]; then
    echo "Committing changes..."
    git commit -m "Add MIT License, update README and requirements for TPU

- Added MIT License
- Updated README with installation instructions and TPU focus
- Updated requirements.txt for TPU support
- Enhanced .gitignore for validation outputs
- Added contribution guidelines"
    echo "Changes committed!"
else
    echo "Skipping commit. You can commit manually later."
fi

echo ""
echo "=========================================="
echo "Ready to push to new repository!"
echo "=========================================="
echo ""
echo "To push to the new repository, run one of:"
echo ""
echo "  # Push current branch to new repo:"
echo "  git push scalesim clean"
echo ""
echo "  # Or push and set as default upstream:"
echo "  git push -u scalesim clean"
echo ""
echo "  # To push all branches:"
echo "  git push scalesim --all"
echo ""
echo "Note: Make sure the repository exists on GitHub first:"
echo "https://github.com/scalesim-project/SCALE-Sim-validation"
echo ""

