# Contribution Guidelines

## Development Process

### 1. Setup

1. **Fork the Repository**:Click the "Fork" button on the [main repository](https://github.com/org/quantitative-trading-system) to create your copy.
2. **Clone Your Fork**:Replace `your-username` with your GitHub handle:

   ```bash
   git clone https://github.com/your-username/quantitative-trading-system.git
   cd quantitative-trading-system
   ```
3. **Set Up a Virtual Environment**:Create and activate a Python virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Linux/macOS
   .venv\Scripts\activate      # Windows
   ```
4. **Install Dependencies**:
   Install development requirements:

   ```bash
   pip install -r requirements-dev.txt
   ```

---

### 2. Branch Management

#### Branch Naming Conventions

Prefix branches with one of the following types:

- `feature/`: New functionality (e.g., `feature/risk-management-module`)
- `bugfix/`: Bug resolution (e.g., `bugfix/order-execution-latency`)
- `docs/`: Documentation updates (e.g., `docs/api-guide`)
- `refactor/`: Code restructuring (e.g., `refactor/backtest-engine`)
- `test/`: Test-related changes (e.g., `test/coverage-improvement`)

#### Workflow Example

```bash
# Create a feature branch
git checkout -b feature/your-feature-name

# Stage and commit changes
git add .
git commit -m "feat(module): brief description of changes"

# Push to your fork
git push origin feature/your-feature-name
```

---

### 3. Commit Guidelines

#### Commit Message Structure

Use [Conventional Commits](https://www.conventionalcommits.org/) format:

```
<type>(<scope>): <description>
```

**Types**:

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code formatting (e.g., whitespace)
- `refactor`: Code restructuring (no functional changes)
- `test`: Test additions/modifications
- `chore`: Maintenance tasks (e.g., dependency updates)

**Examples**:

- Good: `git commit -m "feat(backtest): add Monte Carlo simulation support"`
- Avoid: `git commit -m "fixed some bugs"`

---

### 4. Pull Request (PR) Process

1. **Create a PR**:Target the `main` branch of the original repository.
2. **Use the PR Template**:

   ```markdown
   ## Description
   - **Issue**: Link to related GitHub issue (e.g., Closes #123)
   - **Purpose**: Explain the problem solved or feature added.

   ## Changes
   - Added Monte Carlo simulation engine
   - Optimized data preprocessing pipeline

   ## Testing
   - Added 15 unit tests for simulation logic (95% coverage)
   - Verified against historical SP500 data

   ## Checklist
   - [x] Code follows PEP 8 and Black formatting
   - [x] Unit tests pass (`pytest --cov=src`)
   - [x] Updated `CHANGELOG.md`
   - [x] Documentation reviewed by a peer
   ```
3. **Address Feedback**:

   - Respond to review comments promptly.
   - Update the PR with `git commit --amend` or additional commits.

---

### 5. Development Standards

#### Code Quality

- **Formatting**: Use `black .` to auto-format code.
- **Linting**: Ensure `flake8` passes with no warnings.
- **Type Hints**: Mandatory for all function signatures.
- **Docstrings**: Follow [Google-style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) format.

**Example**:

```python
def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """
    Compute the annualized Sharpe ratio for a returns series.

    Args:
        returns: Daily portfolio returns (e.g., [0.01, -0.005, ...])
        risk_free_rate: Annualized risk-free rate (default: 0.02)

    Returns:
        Annualized Sharpe ratio (float)

    Raises:
        ValueError: If returns series is empty
    """
    if returns.empty:
        raise ValueError("Returns series cannot be empty.")
  
    excess_returns = returns - risk_free_rate / 252
    return excess_returns.mean() / excess_returns.std() * np.sqrt(252)
```

---

### 6. Testing Requirements

#### Test Pyramid

- **Unit Tests**: 70% minimum coverage (core logic only)
- **Integration Tests**: Validate module interactions
- **Performance Tests**: Profile critical paths (e.g., backtesting)

**Run Tests**:

```bash
# Run all tests with coverage
pytest tests/ --cov=src --cov-report=term-missing

# Generate HTML coverage report
pytest --cov=src --cov-report=html
```

---

### 7. Documentation

- **Code Comments**: Explain "why," not "what."
- **API Docs**: Use Sphinx with autodoc (see `/docs`).
- **User Guides**: Update `/docs/user_guides` for new features.

---

### 8. Performance & Security

#### Optimization Guidelines

- Use vectorization over loops in pandas/numpy.
- Prefer `asyncio` for I/O-bound operations.
- Cache expensive computations with `functools.lru_cache`.

#### Security Practices

- Store secrets in environment variables (never in code).
- Sanitize user inputs with `pydantic` models.
- Use `bandit` for static security analysis.

---

### 9. Continuous Integration (CI)

**GitHub Actions Workflows**:

- `build.yml`: Runs tests and linting on PRs.
- `deploy.yml`: Automated PyPI releases on version tags.

---

### 10. Community & Support

- **Discussions**: Use GitHub Discussions for design proposals.
- **Slack**: Join `#quant-trading` channel for real-time chat.
- **Office Hours**: Biweekly Zoom meetings (see [Wiki](https://github.com/.../wiki)).

---

## License & Ethics

- **License**: MIT License (see `LICENSE`).
- **Disclaimer**: This is research software—verify results independently.
- **Ethics**: Never use live trading without thorough backtesting.

---

**Final Checklist Before Submission**:

1. All tests pass locally.
2. Documentation reflects changes.
3. No secrets or sensitive data committed.
4. Branch is rebased on latest `main`.

Thank you for contributing to quantitative finance! 📈🚀
