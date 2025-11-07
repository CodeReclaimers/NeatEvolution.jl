# Setting Up Code Coverage with Codecov

## What Was Added

1. **Updated CI.yaml** - Modified `.github/workflows/CI.yaml` to:
   - Run coverage only on Ubuntu + Julia 1.11 (to avoid redundant uploads)
   - Use `julia-actions/julia-runtest@v1` with coverage enabled
   - Process coverage with `julia-actions/julia-processcoverage@v1`
   - Upload to Codecov with `codecov/codecov-action@v4`

2. **Added .codecov.yml** - Configuration file for Codecov:
   - Set coverage range to 70-100%
   - Ignore test, examples, and docs directories
   - Configure status checks for PRs

3. **Updated README.md** - Added coverage badge

## Setup Steps

### 1. Sign up for Codecov (One-time)

1. Go to https://codecov.io
2. Sign in with your GitHub account
3. Enable the repository: CodeReclaimers/NEAT.jl

### 2. Add Codecov Token to GitHub Secrets

**Option A: Using Codecov App (Recommended - No token needed)**

The Codecov GitHub App automatically handles authentication. If you installed the app, you can skip adding the token.

**Option B: Using Token**

1. Get your Codecov token:
   - Go to https://codecov.io/gh/CodeReclaimers/NEAT.jl/settings
   - Copy the repository upload token

2. Add it to GitHub secrets:
   - Go to https://github.com/CodeReclaimers/NEAT.jl/settings/secrets/actions
   - Click "New repository secret"
   - Name: `CODECOV_TOKEN`
   - Value: [paste your token]
   - Click "Add secret"

### 3. Push Changes

```bash
git add .github/workflows/CI.yaml .codecov.yml README.md docs/CODECOV_SETUP.md
git commit -m "Add code coverage reporting with Codecov"
git push
```

### 4. Verify Coverage

After pushing:
1. Check GitHub Actions tab - CI should run successfully
2. Go to https://codecov.io/gh/CodeReclaimers/NEAT.jl
3. View coverage reports and graphs

## What You'll Get

- **Coverage badge** in README showing overall coverage percentage
- **PR comments** showing coverage changes for each pull request
- **Coverage reports** showing which lines are covered by tests
- **Coverage graphs** showing coverage trends over time
- **File browser** showing coverage for each file

## Troubleshooting

### Coverage not uploading

If coverage upload fails with authentication error:
1. Make sure you've added `CODECOV_TOKEN` to GitHub secrets
2. Or install the Codecov GitHub App (easier)

### Coverage seems low

This is normal initially. To improve:
1. Add more tests for uncovered code
2. Check the coverage report at codecov.io to see what's missing
3. Focus on core functionality (tests/examples are ignored)

### CI failing on nightly Julia

This is expected - nightly builds can be unstable. The workflow is configured to allow nightly failures with `continue-on-error: true`.

## Configuration Details

### Coverage Matrix

Coverage is only computed for:
- OS: Ubuntu (faster, most common)
- Julia: 1.11 (current stable)

This avoids redundant coverage uploads from multiple OS/version combinations.

### Ignored Paths

Per `.codecov.yml`, these paths are excluded from coverage:
- `test/**/*` - Test files themselves
- `examples/**/*` - Example code
- `docs/**/*` - Documentation

This focuses coverage metrics on the actual library code in `src/` and `ext/`.

## Further Customization

Edit `.codecov.yml` to customize:
- Target coverage percentage
- Comment format
- Status check behavior
- Ignored paths

See: https://docs.codecov.com/docs/codecov-yaml
