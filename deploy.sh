#!/bin/bash

# ========================================
# Portfolio Deployment Script for GitHub
# ========================================

echo "🚀 Portfolio Deployment Helper"
echo "================================"
echo ""

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "❌ Error: Git is not installed."
    echo "Please install git first: sudo apt install git"
    exit 1
fi

# Check if already initialized
if [ -d ".git" ]; then
    echo "✅ Git repository already initialized"
else
    echo "📝 Initializing git repository..."
    git init
    git branch -M main
    echo "✅ Git initialized successfully"
fi

# Get GitHub username
echo ""
echo "Please enter your GitHub username:"
read -r github_username

if [ -z "$github_username" ]; then
    echo "❌ Username cannot be empty"
    exit 1
fi

# Set up remote
repo_url="https://github.com/${github_username}/${github_username}.github.io.git"

if git remote | grep -q "origin"; then
    echo "🔄 Remote already exists, updating..."
    git remote set-url origin "$repo_url"
else
    echo "🔗 Adding remote repository..."
    git remote add origin "$repo_url"
fi

echo "✅ Remote set to: $repo_url"
echo ""

# Stage all files
echo "📦 Staging files..."
git add .

# Commit
echo "💾 Creating commit..."
git commit -m "Update: Portfolio changes $(date '+%Y-%m-%d %H:%M:%S')"

# Push
echo ""
echo "🚀 Pushing to GitHub..."
echo "You'll need to enter your GitHub credentials:"
echo "  Username: $github_username"
echo "  Password: Use a Personal Access Token (not your password)"
echo ""

git push -u origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "✨ ================================ ✨"
    echo "🎉 Successfully deployed!"
    echo "✨ ================================ ✨"
    echo ""
    echo "📍 Your portfolio will be live at:"
    echo "   https://${github_username}.github.io"
    echo ""
    echo "⏱️  Note: It may take 2-5 minutes to go live"
    echo ""
    echo "Next steps:"
    echo "  1. Go to: https://github.com/${github_username}/${github_username}.github.io"
    echo "  2. Click Settings → Pages"
    echo "  3. Ensure Source is set to 'main' branch"
    echo "  4. Wait a few minutes and visit your live site!"
else
    echo ""
    echo "❌ Deployment failed"
    echo "Common issues:"
    echo "  - Make sure the repository exists on GitHub"
    echo "  - Use a Personal Access Token instead of password"
    echo "  - Check your internet connection"
    echo ""
    echo "Create token at: https://github.com/settings/tokens"
fi
