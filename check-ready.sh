#!/bin/bash

# Portfolio Deployment Checklist
# Run this before deploying to check if everything is ready

echo "🎯 Portfolio Deployment Checklist"
echo "=================================="
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Initialize counters
ready=0
warnings=0

# Check 1: Git installed
if command -v git &> /dev/null; then
    echo -e "${GREEN}✅${NC} Git is installed"
    ((ready++))
else
    echo -e "${RED}❌${NC} Git is NOT installed (sudo apt install git)"
    ((warnings++))
fi

# Check 2: Files exist
echo ""
echo "📁 Checking files..."
files=("index.html" "style.css" "script.js" "README.md")
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✅${NC} $file exists"
        ((ready++))
    else
        echo -e "${RED}❌${NC} $file is missing"
        ((warnings++))
    fi
done

# Check 3: Personalization check
echo ""
echo "👤 Checking personalization..."

if grep -q "Your Name" index.html; then
    echo -e "${YELLOW}⚠️${NC}  'Your Name' placeholder found in index.html"
    echo "   → Update your actual name"
    ((warnings++))
else
    echo -e "${GREEN}✅${NC} Name appears to be customized"
    ((ready++))
fi

if grep -q "your.email@example.com" index.html; then
    echo -e "${YELLOW}⚠️${NC}  Default email found in index.html"
    echo "   → Update with your real email"
    ((warnings++))
else
    echo -e "${GREEN}✅${NC} Email appears to be customized"
    ((ready++))
fi

if grep -q "yourusername" index.html; then
    echo -e "${YELLOW}⚠️${NC}  'yourusername' placeholder found"
    echo "   → Update with your GitHub/social media usernames"
    ((warnings++))
else
    echo -e "${GREEN}✅${NC} Social links appear to be customized"
    ((ready++))
fi

# Check 4: Git status
echo ""
echo "🔍 Git status..."
if [ -d ".git" ]; then
    echo -e "${GREEN}✅${NC} Git repository initialized"
    ((ready++))
    
    if git remote | grep -q "origin"; then
        remote_url=$(git remote get-url origin)
        echo -e "${GREEN}✅${NC} Remote configured: $remote_url"
        ((ready++))
    else
        echo -e "${YELLOW}⚠️${NC}  No remote repository configured"
        echo "   → Run ./deploy.sh or add remote manually"
        ((warnings++))
    fi
else
    echo -e "${YELLOW}⚠️${NC}  Git not initialized"
    echo "   → Run: git init"
    ((warnings++))
fi

# Summary
echo ""
echo "=================================="
echo "📊 Summary:"
echo "   Ready: $ready items"
echo "   Warnings: $warnings items"
echo ""

if [ $warnings -eq 0 ]; then
    echo -e "${GREEN}🎉 Your portfolio is ready to deploy!${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Create repository on GitHub: yourusername.github.io"
    echo "2. Run: ./deploy.sh"
    echo "3. Enable GitHub Pages in repository settings"
else
    echo -e "${YELLOW}⚠️  Please address the warnings above before deploying${NC}"
    echo ""
    echo "See QUICKSTART.md for detailed instructions"
fi

echo ""
