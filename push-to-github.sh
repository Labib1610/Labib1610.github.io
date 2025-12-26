#!/bin/bash

# Simple Push Script for Labib1610.github.io
# This script will push your portfolio to GitHub

echo "═══════════════════════════════════════════════════════════════"
echo "     Pushing Data Science Portfolio to GitHub"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Repository: https://github.com/Labib1610/Labib1610.github.io"
echo ""
echo "📝 You will be prompted for:"
echo "   Username: Labib1610"
echo "   Password: [Your Personal Access Token]"
echo ""
echo "🔑 Don't have a token? Create one at:"
echo "   https://github.com/settings/tokens"
echo "   (Select 'repo' permissions)"
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Navigate to portfolio directory
cd "/mnt/wwn-0x500a0751e14d807e-part2/Labib/Labib Folder/Labib/UIU/my-portfolio"

# Push to GitHub
echo "🚀 Pushing to GitHub..."
git push -u origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "✅ SUCCESS! Portfolio deployed to GitHub!"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""
    echo "🌐 Your portfolio will be live at:"
    echo "   https://Labib1610.github.io"
    echo ""
    echo "⏱️  Wait 2-5 minutes for GitHub Pages to deploy"
    echo ""
    echo "Next steps:"
    echo "1. Go to: https://github.com/Labib1610/Labib1610.github.io"
    echo "2. Click Settings → Pages"
    echo "3. Ensure Source: main branch, / (root)"
    echo "4. Wait a few minutes"
    echo "5. Visit: https://Labib1610.github.io"
    echo ""
else
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "❌ Push failed!"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""
    echo "Common issues:"
    echo ""
    echo "1. Using password instead of token:"
    echo "   → Create token: https://github.com/settings/tokens"
    echo "   → Use TOKEN as password (not your GitHub password)"
    echo ""
    echo "2. Repository doesn't exist:"
    echo "   → Create it at: https://github.com/new"
    echo "   → Name: Labib1610.github.io"
    echo "   → Make it PUBLIC"
    echo ""
    echo "3. Already pushed before:"
    echo "   → Just run: git push"
    echo ""
    echo "Need help? Check: DEPLOYMENT-GUIDE-LABIB.txt"
    echo ""
fi
