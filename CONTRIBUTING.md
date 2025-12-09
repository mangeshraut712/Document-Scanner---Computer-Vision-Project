# Contributing to Document Scanner

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## 🚀 Getting Started

1. **Fork the repository**
   ```bash
   git clone https://github.com/mangeshraut712/Document-Scanner---Computer-Vision-Project.git
   cd Document-Scanner---Computer-Vision-Project
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Follow the coding style guidelines below
   - Add tests for new functionality
   - Update documentation as needed

4. **Test your changes**
   ```bash
   pytest tests/ -v
   ```

5. **Commit and push**
   ```bash
   git commit -m "Add feature: brief description"
   git push origin feature/your-feature-name
   ```

6. **Create a Pull Request**

## 📝 Coding Standards

### MATLAB
- Use camelCase for variables and functions
- Use PascalCase for classes
- Add comprehensive comments
- Follow MATLAB style guidelines

### Python
- Follow PEP 8 style guide
- Use snake_case for functions and variables
- Use PascalCase for classes
- Add type hints
- Write docstrings for all functions

### JavaScript
- Use camelCase for variables and functions
- Use const/let, not var
- Add JSDoc comments
- Follow modern ES6+ syntax

## 🧪 Testing

All new features should include tests:

```python
def test_new_feature():
    scanner = DocumentScanner('test_image.png')
    result = scanner.new_feature()
    assert result is not None
```

## 📚 Documentation

- Update README.md for user-facing changes
- Add inline comments for complex algorithms
- Update docstrings when modifying functions

## 🎯 Areas for Contribution

We especially welcome contributions in:

1. **Automatic Corner Detection**
2. **Performance Optimization**
3. **Additional Features** (OCR, batch processing)
4. **Testing** (more test cases, benchmarks)
5. **Documentation** (examples, tutorials)

## 🤝 Code Review Process

1. All PRs require at least one review
2. CI tests must pass
3. Code coverage should not decrease
4. Documentation must be updated

## 📞 Questions?

- Open a discussion on GitHub
- Check existing issues and PRs

Thank you for contributing! 🎉
