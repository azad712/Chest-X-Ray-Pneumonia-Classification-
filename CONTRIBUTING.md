# Contributing to Pneumonia Classification Project

Thank you for your interest in contributing to this project! We welcome contributions from the community.

## 🤝 How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Your environment (OS, Python version, TensorFlow version)
- Error messages or logs

### Suggesting Enhancements

We welcome suggestions for:
- New features
- Performance improvements
- Better documentation
- Code optimizations
- Additional evaluation metrics

### Pull Requests

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Follow the existing code style
   - Add comments for complex logic
   - Update documentation if needed

4. **Test your changes**
   ```bash
   # Run training (if applicable)
   python train_pneumonia_classifier.py
   
   # Test predictions
   python predict_pneumonia.py --model models/best_model.h5 --image test.jpg
   
   # Run evaluation
   python evaluate_pneumonia.py --model models/best_model.h5
   ```

5. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add: Brief description of your changes"
   ```

6. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create a Pull Request**
   - Provide a clear description
   - Reference any related issues
   - Include screenshots if applicable

## 📝 Code Style Guidelines

### Python Code
- Follow PEP 8 style guide
- Use meaningful variable names
- Add docstrings to functions and classes
- Keep functions focused and concise
- Use type hints where appropriate

Example:
```python
def preprocess_image(image_path: str, target_size: tuple = (224, 224)) -> np.ndarray:
    """
    Preprocess an image for model input.
    
    Args:
        image_path: Path to the image file
        target_size: Target size for resizing (height, width)
    
    Returns:
        Preprocessed image array
    """
    # Implementation here
    pass
```

### Documentation
- Update README.md for new features
- Add inline comments for complex logic
- Include usage examples
- Document any new dependencies

### Commit Messages
Use clear, descriptive commit messages:
- `Add: New feature description`
- `Fix: Bug description`
- `Update: What was updated`
- `Refactor: What was refactored`
- `Docs: Documentation changes`

## 🧪 Testing

Before submitting a PR, ensure:
- [ ] Code runs without errors
- [ ] Training completes successfully (if modified)
- [ ] Predictions work correctly
- [ ] Evaluation produces expected results
- [ ] No breaking changes to existing functionality
- [ ] Documentation is updated

## 🎯 Areas for Contribution

We especially welcome contributions in:

### 1. Model Improvements
- Experiment with different architectures
- Hyperparameter tuning
- Advanced data augmentation techniques
- Ensemble methods

### 2. Features
- Web interface for predictions
- REST API for model serving
- Batch processing improvements
- Real-time inference optimization

### 3. Evaluation
- Additional metrics (Specificity, NPV, etc.)
- ROC curve visualization
- Confusion matrix improvements
- Per-class analysis

### 4. Documentation
- Tutorial notebooks
- Video tutorials
- Better examples
- Troubleshooting guide

### 5. Dataset
- Data preprocessing improvements
- Class imbalance handling
- Cross-validation implementation
- Data quality checks

### 6. Deployment
- Docker containerization
- Cloud deployment guides (AWS, GCP, Azure)
- Mobile deployment (TensorFlow Lite)
- ONNX conversion

## 🔍 Code Review Process

1. Maintainers will review your PR
2. Feedback will be provided if changes are needed
3. Once approved, your PR will be merged
4. You'll be added to the contributors list!

## 📋 Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/pneumonia-classification.git
cd pneumonia-classification

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install pytest black flake8 mypy

# Run code formatting
black *.py

# Run linting
flake8 *.py
```

## 🐛 Debugging Tips

- Use `print()` statements or logging for debugging
- Test with small datasets first
- Check TensorFlow/GPU compatibility
- Verify file paths are correct
- Monitor memory usage for large datasets

## 📞 Getting Help

- Create an issue for questions
- Check existing issues and PRs
- Review documentation thoroughly
- Provide detailed information when asking for help

## 🏆 Recognition

Contributors will be:
- Listed in the README
- Mentioned in release notes
- Credited in documentation

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 🙏 Thank You!

Every contribution, no matter how small, is valuable and appreciated. Thank you for helping improve this project!

---

**Happy Contributing! 🚀**