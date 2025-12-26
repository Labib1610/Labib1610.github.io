# 🚀 Professional Portfolio Website

A modern, responsive portfolio website showcasing your projects, skills, and professional experience. Built with HTML5, CSS3, and JavaScript with stunning animations and a professional design.

![Portfolio Preview](https://img.shields.io/badge/Status-Ready-success)
![HTML5](https://img.shields.io/badge/HTML5-E34F26?logo=html5&logoColor=white)
![CSS3](https://img.shields.io/badge/CSS3-1572B6?logo=css3&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?logo=javascript&logoColor=black)

## ✨ Features

- 🎨 **Modern Design** - Clean, professional UI with gradient accents
- 📱 **Fully Responsive** - Works perfectly on all devices
- ⚡ **Smooth Animations** - Engaging scroll effects and transitions
- 🌈 **Animated Background** - Dynamic gradient circles
- 💼 **Project Showcase** - Beautiful cards to display your work
- 🛠️ **Skills Section** - Highlight your technical expertise
- 📧 **Contact Section** - Easy ways for people to reach you
- 🔝 **Scroll to Top** - Convenient navigation button
- ⌨️ **Typing Animation** - Dynamic text effect in hero section

## 🎯 Customization Guide

### 1. Personal Information

**In `index.html`**, update the following:

- **Line 6**: Change page title
  ```html
  <title>Your Name - Professional Portfolio</title>
  ```

- **Lines 35-38**: Update your name and bio
  ```html
  <h1 class="hero-name">
      <span class="highlight">Your Actual Name</span>
  </h1>
  ```

- **Line 41**: Add your description
- **Lines 377-381**: Update contact information (email, phone, location)
- **Lines 391-402**: Add your social media links

### 2. Skills & Technologies

**In `script.js`** (lines 8-13), customize the typing animation roles:
```javascript
const roles = [
    "Your Title Here",
    "Your Specialty",
    "Your Skill",
    "Your Passion"
];
```

**In `index.html`** (Skills Section), update your technical skills.

### 3. Projects

**In `index.html`** (lines 199-345), customize each project:
- Update project titles
- Change descriptions
- Modify technology tags
- Add GitHub repository links
- Add live demo links

### 4. Color Scheme (Optional)

**In `style.css`** (lines 6-15), modify the color variables:
```css
:root {
    --primary-color: #6366f1;  /* Main brand color */
    --secondary-color: #8b5cf6; /* Secondary color */
    --accent-color: #ec4899;    /* Accent color */
}
```

## 📦 Project Structure

```
my-portfolio/
│
├── index.html          # Main HTML file
├── style.css           # Stylesheet with animations
├── script.js           # JavaScript functionality
└── README.md           # This file
```

## 🌐 Deployment to GitHub Pages

### Step 1: Create a GitHub Repository

1. Go to [GitHub](https://github.com) and log in
2. Click the **+** icon (top right) → **New repository**
3. Name your repository: `your-username.github.io` (replace with your actual GitHub username)
   - Example: If your username is "johndoe", name it: `johndoe.github.io`
4. Keep it **Public**
5. **Don't** initialize with README (you already have one)
6. Click **Create repository**

### Step 2: Initialize Git in Your Project

Open terminal in your project folder and run:

```bash
# Navigate to your portfolio folder
cd /mnt/wwn-0x500a0751e14d807e-part2/Labib/Labib\ Folder/Labib/UIU/my-portfolio

# Initialize git repository
git init

# Add all files
git add .

# Commit your changes
git commit -m "Initial commit: Professional portfolio"

# Rename branch to main (if needed)
git branch -M main
```

### Step 3: Connect to GitHub and Push

```bash
# Add your GitHub repository as remote
# Replace YOUR-USERNAME with your actual GitHub username
git remote add origin https://github.com/YOUR-USERNAME/YOUR-USERNAME.github.io.git

# Push your code to GitHub
git push -u origin main
```

**Important:** You'll be prompted for your GitHub credentials:
- Username: Your GitHub username
- Password: Use a [Personal Access Token](https://github.com/settings/tokens) (not your password)

### Step 4: Enable GitHub Pages

1. Go to your repository on GitHub
2. Click **Settings** (top menu)
3. Click **Pages** (left sidebar)
4. Under **Source**, select:
   - Branch: `main`
   - Folder: `/ (root)`
5. Click **Save**

### Step 5: Access Your Website

Your portfolio will be live at: `https://your-username.github.io`

**Note:** It may take 2-5 minutes for the site to go live.

## 🔄 Updating Your Portfolio

Whenever you make changes:

```bash
# Stage all changes
git add .

# Commit with a message
git commit -m "Update: describe your changes"

# Push to GitHub
git push
```

Your site will automatically update within minutes!

## 📱 Testing Locally

Before deploying, test your portfolio locally:

1. **Option 1:** Open `index.html` directly in your browser

2. **Option 2:** Use a local server (recommended)
   ```bash
   # If you have Python installed:
   python -m http.server 8000
   # Then visit: http://localhost:8000
   ```

3. **Option 3:** Use VS Code Live Server extension

## 🎨 Customization Tips

### Adding Your Photo
1. Add an image file to your project folder
2. In `index.html`, replace the profile icon section with:
   ```html
   <img src="your-photo.jpg" alt="Your Name" class="profile-photo">
   ```
3. Add CSS for `.profile-photo` class

### Adding More Projects
Copy one of the existing `.project-card` divs and customize it with your new project details.

### Changing Fonts
The portfolio uses Google Fonts (Poppins). To change:
1. Visit [Google Fonts](https://fonts.google.com)
2. Select your preferred font
3. Replace the font link in `index.html`
4. Update the `font-family` in `style.css`

## 🐛 Troubleshooting

### Site Not Loading?
- Wait 5-10 minutes after initial deployment
- Check GitHub Pages settings are correct
- Ensure your repository is public
- Clear browser cache and try again

### Git Push Issues?
- Make sure you're using a Personal Access Token, not your password
- Verify the remote URL: `git remote -v`
- Check your internet connection

### Links Not Working?
- Verify all `href` attributes are correct
- For external links, include `https://`
- For internal links, use `#section-id`

## 📚 Resources

- [GitHub Pages Documentation](https://docs.github.com/en/pages)
- [Git Basics](https://git-scm.com/book/en/v2/Getting-Started-Git-Basics)
- [HTML/CSS/JS Reference](https://developer.mozilla.org/en-US/)
- [Font Awesome Icons](https://fontawesome.com/icons)

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🤝 Contributing

Feel free to fork this project and customize it for your own use!

## 📧 Support

If you have questions or need help:
- Open an issue on GitHub
- Check existing issues for solutions
- Refer to the documentation above

---

**Made with ❤️ and lots of coffee** ☕

Happy coding! 🚀
