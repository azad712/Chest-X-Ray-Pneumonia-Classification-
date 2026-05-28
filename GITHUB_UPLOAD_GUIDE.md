# 📤 GitHub Upload Guide

Complete step-by-step guide to upload your Pneumonia Classification project to GitHub.

## 📋 Pre-Upload Checklist

Before uploading, ensure you have:

- [x] ✅ `.gitignore` file created
- [x] ✅ `LICENSE` file added
- [x] ✅ `README.md` updated with badges and instructions
- [x] ✅ `CONTRIBUTING.md` created
- [x] ✅ All code files are present
- [x] ✅ `requirements.txt` is up to date
- [ ] ⚠️ Large files excluded (dataset, model .h5 files)
- [ ] ⚠️ GitHub account created
- [ ] ⚠️ Git installed on your system

---

## 🚀 Method 1: Using Git Command Line (Recommended)

### Step 1: Install Git

**Windows:**
```bash
# Download from: https://git-scm.com/download/win
# Run the installer and follow the prompts
```

**Verify installation:**
```bash
git --version
```

### Step 2: Configure Git

```bash
# Set your name
git config --global user.name "Your Name"

# Set your email (use your GitHub email)
git config --global user.email "your.email@example.com"

# Verify configuration
git config --list
```

### Step 3: Initialize Repository

```bash
# Navigate to your project directory
cd "C:\Users\Admin\Desktop\CMRIT_Content\STTP Medical AI\Chest x ray pneumonia"

# Initialize git repository
git init

# Check status
git status
```

### Step 4: Add Files

```bash
# Add all files (respects .gitignore)
git add .

# Check what will be committed
git status

# You should see:
# - Python files (.py)
# - Documentation files (.md)
# - requirements.txt
# - LICENSE
# - Small model metadata files (history.json, class_names.json, training_history.png)
#
# You should NOT see:
# - chest_xray/ folder
# - .h5 model files
# - __pycache__/
```

### Step 5: Create First Commit

```bash
git commit -m "Initial commit: Pneumonia classification with MobileNetV2

- Add training, prediction, and evaluation scripts
- Add comprehensive documentation
- Add MIT license
- Include training results and metadata
- Exclude large files (dataset and models)"
```

### Step 6: Create GitHub Repository

1. Go to [GitHub](https://github.com)
2. Click the **"+"** icon (top right) → **"New repository"**
3. Fill in details:
   - **Repository name**: `pneumonia-classification` (or your choice)
   - **Description**: `Chest X-ray Pneumonia Classification using MobileNetV2 and Transfer Learning`
   - **Visibility**: Choose Public or Private
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)
4. Click **"Create repository"**

### Step 7: Connect and Push

```bash
# Add remote repository (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/pneumonia-classification.git

# Verify remote
git remote -v

# Rename branch to main (if needed)
git branch -M main

# Push to GitHub
git push -u origin main
```

**If prompted for credentials:**
- Username: Your GitHub username
- Password: Use a **Personal Access Token** (not your password)
  - Generate token at: Settings → Developer settings → Personal access tokens → Tokens (classic)
  - Select scopes: `repo` (full control)

### Step 8: Verify Upload

1. Go to your GitHub repository URL
2. Verify all files are present
3. Check that README displays correctly
4. Confirm large files are NOT uploaded

---

## 🖥️ Method 2: Using GitHub Desktop (Easier)

### Step 1: Install GitHub Desktop

1. Download from: https://desktop.github.com/
2. Install and sign in with your GitHub account

### Step 2: Add Repository

1. Open GitHub Desktop
2. Click **"File"** → **"Add Local Repository"**
3. Browse to: `C:\Users\Admin\Desktop\CMRIT_Content\STTP Medical AI\Chest x ray pneumonia`
4. Click **"Add Repository"**

If it says "This directory does not appear to be a Git repository":
- Click **"Create a repository"**
- Uncheck "Initialize with README" (we already have one)
- Click **"Create Repository"**

### Step 3: Review Changes

1. You'll see all files in the "Changes" tab
2. Verify that:
   - ✅ Python files are listed
   - ✅ Documentation files are listed
   - ❌ chest_xray/ folder is NOT listed
   - ❌ .h5 files are NOT listed

### Step 4: Commit

1. Add commit message: `Initial commit: Pneumonia classification project`
2. Add description (optional)
3. Click **"Commit to main"**

### Step 5: Publish to GitHub

1. Click **"Publish repository"**
2. Choose repository name: `pneumonia-classification`
3. Add description
4. Choose Public or Private
5. Uncheck "Keep this code private" if you want it public
6. Click **"Publish Repository"**

### Step 6: Verify

1. Click **"View on GitHub"** button
2. Verify all files uploaded correctly

---

## 💻 Method 3: Using VS Code (Integrated)

### Step 1: Open Project in VS Code

```bash
# Open VS Code in project directory
cd "C:\Users\Admin\Desktop\CMRIT_Content\STTP Medical AI\Chest x ray pneumonia"
code .
```

### Step 2: Initialize Repository

1. Click **Source Control** icon (left sidebar, looks like a branch)
2. Click **"Initialize Repository"**
3. Select the current folder

### Step 3: Stage and Commit

1. You'll see all changed files
2. Click **"+"** next to "Changes" to stage all files
3. Enter commit message: `Initial commit: Pneumonia classification project`
4. Click **✓** (checkmark) to commit

### Step 4: Publish to GitHub

1. Click **"Publish to GitHub"** button
2. Choose repository name
3. Select Public or Private
4. Click **"Publish"**
5. Sign in to GitHub if prompted

---

## 📦 Uploading Large Model Files (Optional)

Since model files are too large for GitHub, you have options:

### Option 1: Git LFS (Large File Storage)

```bash
# Install Git LFS
git lfs install

# Track .h5 files
git lfs track "*.h5"

# Add .gitattributes
git add .gitattributes

# Add and commit model files
git add models/
git commit -m "Add trained models via Git LFS"
git push
```

**Note**: Free GitHub accounts have LFS limits (1GB storage, 1GB bandwidth/month)

### Option 2: External Hosting (Recommended)

Upload models to:

1. **Google Drive**
   - Upload .h5 files
   - Get shareable link
   - Add link to README

2. **Hugging Face**
   - Create account at huggingface.co
   - Upload model
   - Add link to README

3. **Dropbox**
   - Upload files
   - Get shareable link
   - Add link to README

4. **GitHub Releases**
   - Create a release
   - Attach .h5 files (up to 2GB per file)
   - Link in README

---

## 🔄 Making Updates After Initial Upload

### Update Files

```bash
# Make your changes to files

# Check what changed
git status

# Stage changes
git add .

# Commit with descriptive message
git commit -m "Update: Description of changes"

# Push to GitHub
git push
```

### Common Update Commands

```bash
# Add new file
git add new_file.py
git commit -m "Add: New feature description"
git push

# Update existing file
git add modified_file.py
git commit -m "Fix: Bug description"
git push

# Delete file
git rm old_file.py
git commit -m "Remove: Obsolete file"
git push
```

---

## 🎨 Enhancing Your Repository

### Add Repository Topics

1. Go to your repository on GitHub
2. Click ⚙️ (gear icon) next to "About"
3. Add topics: `deep-learning`, `medical-imaging`, `pneumonia-detection`, `tensorflow`, `mobilenetv2`, `computer-vision`, `healthcare`, `machine-learning`

### Create Repository Description

Add a short description:
```
🏥 Automated pneumonia detection from chest X-rays using MobileNetV2 and transfer learning. Achieves 95.4% training accuracy with comprehensive evaluation metrics.
```

### Add Website Link

Link to:
- Documentation
- Demo (if you create one)
- Dataset source

### Enable GitHub Pages (Optional)

1. Settings → Pages
2. Source: Deploy from branch
3. Branch: main, folder: /docs (if you create docs)

---

## 🐛 Troubleshooting

### Problem: "Large files detected"

**Solution:**
```bash
# Remove large files from staging
git rm --cached models/*.h5
git commit -m "Remove large model files"

# Update .gitignore to exclude them
echo "*.h5" >> .gitignore
git add .gitignore
git commit -m "Update .gitignore"
git push
```

### Problem: "Authentication failed"

**Solution:**
- Use Personal Access Token instead of password
- Generate at: GitHub Settings → Developer settings → Personal access tokens

### Problem: "Permission denied"

**Solution:**
```bash
# Check remote URL
git remote -v

# If using HTTPS, switch to SSH or use token
git remote set-url origin https://YOUR_TOKEN@github.com/YOUR_USERNAME/repo.git
```

### Problem: Files not being ignored

**Solution:**
```bash
# Remove cached files
git rm -r --cached .

# Re-add files (will respect .gitignore)
git add .
git commit -m "Fix: Apply .gitignore rules"
git push
```

---

## ✅ Post-Upload Checklist

After uploading, verify:

- [ ] ✅ Repository is accessible
- [ ] ✅ README displays correctly with badges
- [ ] ✅ All code files are present
- [ ] ✅ LICENSE file is visible
- [ ] ✅ .gitignore is working (no large files)
- [ ] ✅ Repository description is set
- [ ] ✅ Topics are added
- [ ] ✅ Links in README work
- [ ] ✅ Code is properly formatted
- [ ] ✅ No sensitive information exposed

---

## 🎯 Next Steps

1. **Add a Release**
   - Tag your version (v1.0.0)
   - Attach model files
   - Write release notes

2. **Create Issues**
   - Document known issues
   - Plan future enhancements

3. **Set Up CI/CD** (Advanced)
   - GitHub Actions for testing
   - Automated code quality checks

4. **Share Your Project**
   - Post on LinkedIn
   - Share on Twitter
   - Add to your portfolio

---

## 📞 Need Help?

- **Git Documentation**: https://git-scm.com/doc
- **GitHub Guides**: https://guides.github.com/
- **GitHub Support**: https://support.github.com/

---

## 🎉 Congratulations!

Your project is now on GitHub! 🚀

**Repository URL Format:**
```
https://github.com/YOUR_USERNAME/pneumonia-classification
```

Share it with the world! 🌍