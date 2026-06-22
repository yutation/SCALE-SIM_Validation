# Setup Guide: Pushing to GitHub

This guide will help you push the SCALE-Sim Validation framework to the new GitHub repository.

## Repository Information

- **New Repository**: https://github.com/scalesim-project/SCALE-Sim-validation
- **Current Branch**: `clean`
- **License**: MIT License (already added)

## Prerequisites

1. **Create the GitHub Repository** (if not already created):
   - Go to: https://github.com/scalesim-project
   - Create a new repository named `SCALE-Sim-validation`
   - **Important**: Do NOT initialize it with README, .gitignore, or license (we already have these)

2. **GitHub Authentication**: Ensure you have access to push to the scalesim-project organization

## Quick Setup (Automated)

Run the provided setup script:

```bash
cd /home/Owner/work/SCALE-Sim/validation
./push_to_new_repo.sh
```

This script will:
- Show current git status
- Add the new remote repository
- Optionally commit your changes
- Show push commands

## Manual Setup (Step-by-Step)

### Step 1: Add the New Remote

```bash
cd /home/Owner/work/SCALE-Sim/validation
git remote add scalesim https://github.com/scalesim-project/SCALE-Sim-validation.git
```

Verify remotes:
```bash
git remote -v
```

You should see:
- `origin`: Your current repository
- `scalesim`: The new repository

### Step 2: Commit Your Changes

Check what needs to be committed:
```bash
git status
```

Stage and commit the changes:
```bash
git add LICENSE README.md .gitignore requirements.txt
git commit -m "Add MIT License, update README and requirements for TPU

- Added MIT License
- Updated README with installation instructions and TPU focus
- Updated requirements.txt for TPU support
- Enhanced .gitignore for validation outputs
- Added contribution guidelines"
```

### Step 3: Push to the New Repository

Push the current branch:
```bash
git push scalesim clean
```

Or push and set as default upstream:
```bash
git push -u scalesim clean
```

To push all branches:
```bash
git push scalesim --all
```

### Step 4: Verify the Push

Visit the repository:
https://github.com/scalesim-project/SCALE-Sim-validation

You should see:
- ✅ README.md displayed on the main page
- ✅ MIT License badge
- ✅ All source files
- ✅ Requirements.txt

## What's Included

### Files Added/Modified:
- ✅ **LICENSE**: MIT License file
- ✅ **README.md**: Comprehensive documentation
- ✅ **requirements.txt**: TPU-focused dependencies
- ✅ **.gitignore**: Updated to ignore verification outputs

### Files Excluded (via .gitignore):
- Trace directories (`traces/`, `trace/`)
- Verification result directories (`*_verification_results/`)
- CSV output files (except requirements.txt)
- Python cache files (`__pycache__/`)

## Setting Default Remote (Optional)

If you want to make the new repository your default remote:

```bash
# Rename old origin to backup
git remote rename origin origin-backup

# Rename scalesim to origin
git remote rename scalesim origin

# Set upstream for current branch
git branch --set-upstream-to=origin/clean clean
```

## Troubleshooting

### Authentication Issues

If you encounter authentication issues:

1. **Personal Access Token** (Recommended):
   ```bash
   git remote set-url scalesim https://YOUR_TOKEN@github.com/scalesim-project/SCALE-Sim-validation.git
   ```

2. **SSH** (Alternative):
   ```bash
   git remote set-url scalesim git@github.com:scalesim-project/SCALE-Sim-validation.git
   ```

### Repository Already Exists Error

If you see "repository already exists with content":
```bash
# Pull first, then push
git pull scalesim clean --allow-unrelated-histories
git push scalesim clean
```

### Permission Denied

Ensure you have:
- Write access to the scalesim-project organization
- Correct GitHub credentials configured
- Two-factor authentication tokens (if enabled)

## Post-Push Tasks

After successfully pushing:

1. **Set Repository Description** on GitHub:
   - "Validation framework for SCALE-Sim: Profile and validate neural network operations on TPU"

2. **Add Topics/Tags**:
   - `neural-networks`
   - `validation`
   - `tpu`
   - `jax`
   - `performance-testing`
   - `scale-sim`

3. **Configure Branch Protection** (Optional):
   - Require pull request reviews
   - Enable status checks

4. **Update Repository Settings**:
   - Enable Issues (for bug reports)
   - Enable Discussions (for Q&A)
   - Set default branch to `clean`

## Next Steps

1. Update any documentation that references the old repository
2. Notify team members of the new repository location
3. Archive or redirect the old repository (if applicable)
4. Set up CI/CD workflows (if needed)

## Support

For issues with this setup, contact the SCALE-Sim project maintainers or open an issue at:
https://github.com/scalesim-project/SCALE-Sim-validation/issues

