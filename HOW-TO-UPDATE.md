# Portfolio Update Guide

## Overview
Your portfolio now has 4 major new features, all in dummy mode ready for you to customize:

1. **Background Image Slideshow** - Auto-rotating tech images
2. **Featured Projects Carousel** - For ongoing projects
3. **Completed Projects Grid** - For finished work
4. **Notes System** - With filtering and modal viewing

---

## 1. Background Slideshow

### How to Add New Background Images

**Location in HTML:** Lines 30-40 in `index.html`

```html
<div class="bg-slideshow">
    <div class="slide active" style="background-image: url('YOUR-IMAGE-URL')"></div>
    <!-- Add more slides here -->
</div>
```

### Steps:
1. Find images on [Unsplash](https://unsplash.com) or upload your own
2. Copy the image URL
3. Add a new line: `<div class="slide" style="background-image: url('YOUR-URL')"></div>`
4. Can add unlimited images - they rotate every 6 seconds

### Best Practices:
- Use high-quality images (1920x1080 or higher)
- Choose tech/data science themed images
- Images are shown at 15% opacity (darker background)

---

## 2. Featured Projects (Carousel)

### How to Add Featured Projects

**Location in HTML:** Lines 165-280 in `index.html`

Current structure (3 projects):
```html
<div class="featured-project-slide">
    <div class="featured-project-card">
        <div class="featured-project-image">
            <img src="YOUR-IMAGE" alt="Project">
            <div class="project-status">
                <span class="status-badge working">
                    <i class="fas fa-code"></i> In Progress
                </span>
            </div>
        </div>
        <div class="featured-project-content">
            <h3>Your Project Title</h3>
            <p>Project description...</p>
            <div class="project-progress">
                <div class="progress-bar">
                    <div class="progress-fill" style="width: 65%"></div>
                </div>
                <p class="progress-text">65% Complete</p>
            </div>
        </div>
    </div>
</div>
```

### Steps to Add New Featured Project:
1. Copy the entire `<div class="featured-project-slide">` block
2. Paste after the last slide
3. Update:
   - Image URL
   - Project title
   - Description
   - Progress percentage (both in `style="width: XX%"` and text)
   - Status badge (working/testing/deploying)

### Steps to Update Progress:
1. Find your project's `progress-fill` div
2. Change `style="width: 65%"` to your current progress
3. Update the text below to match

### Don't Forget:
- Add a new dot: `<div class="dot"></div>` in the carousel-dots section
- Carousel auto-advances every 8 seconds

---

## 3. Completed Projects (Grid)

### How to Add Completed Projects

**Location in HTML:** Lines 285-440 in `index.html`

Current structure (6 projects):
```html
<div class="completed-project-card">
    <div class="project-header">
        <div class="project-icon-wrapper">
            <i class="fas fa-robot"></i>
        </div>
        <div class="completion-badge">
            <i class="fas fa-check-circle"></i> Completed
        </div>
    </div>
    <h3>Project Title</h3>
    <p>Project description...</p>
    <div class="project-links">
        <a href="#" class="btn btn-primary">
            <i class="fab fa-github"></i> GitHub
        </a>
        <a href="#" class="btn btn-secondary">
            <i class="fas fa-file-pdf"></i> Paper
        </a>
        <a href="#" class="btn btn-accent">
            <i class="fas fa-external-link-alt"></i> Demo
        </a>
    </div>
</div>
```

### Steps to Add New Completed Project:
1. Copy entire `<div class="completed-project-card">` block
2. Paste in the grid
3. Update:
   - Icon class (e.g., `fa-robot`, `fa-brain`, `fa-chart-line`)
   - Project title
   - Description
   - GitHub link
   - Paper link (if available)
   - Demo link (if available)

### Link Types Available:
- GitHub: `<i class="fab fa-github"></i>`
- Paper/PDF: `<i class="fas fa-file-pdf"></i>`
- Demo: `<i class="fas fa-external-link-alt"></i>`
- Dataset: `<i class="fas fa-database"></i>`
- Documentation: `<i class="fas fa-book"></i>`

### Removing Links:
If a project doesn't have a paper or demo, just delete that `<a>` tag.

---

## 4. Notes System

### How to Add External Notes (Google Drive)

**Location in HTML:** Lines 445-600 in `index.html`

For notes stored externally:
```html
<div class="note-card" data-category="ml">
    <div class="note-header">
        <div class="note-icon">
            <i class="fas fa-brain"></i>
        </div>
        <span class="note-type">External</span>
    </div>
    <h3>Note Title</h3>
    <p>Brief description...</p>
    <div class="note-meta">
        <span><i class="far fa-calendar"></i> Jan 2024</span>
        <span><i class="fas fa-tag"></i> Machine Learning</span>
    </div>
    <a href="YOUR-GOOGLE-DRIVE-LINK" class="note-link" target="_blank">
        <i class="fas fa-external-link-alt"></i> Open Note
    </a>
</div>
```

### How to Add Embedded Notes (View in Portfolio)

For notes displayed in modal:
```html
<div class="note-card" data-category="deep-learning" data-content="YOUR-CONTENT-HERE">
    <!-- Same structure but with view-note-btn -->
    <button class="note-link view-note-btn">
        <i class="fas fa-eye"></i> View Note
    </button>
</div>
```

### Steps to Add New Note:
1. Copy a note card block
2. Update:
   - `data-category`: ml, deep-learning, nlp, or tutorials
   - Icon: `fa-brain`, `fa-network-wired`, `fa-comment-dots`, etc.
   - Note type: External or Embedded
   - Title and description
   - Date and tags
   - Link (Google Drive) or button (embedded)

### For Embedded Notes:
Add `data-content` attribute with HTML:
```html
data-content="
    <h2>Note Title</h2>
    <h3>Section 1</h3>
    <p>Your content...</p>
    <pre><code>Your code here</code></pre>
"
```

### Categories Available:
- `all` - Shows all notes
- `ml` - Machine Learning
- `deep-learning` - Deep Learning
- `nlp` - Natural Language Processing
- `tutorials` - Step-by-step guides

### Adding New Category:
1. Add filter button:
```html
<button class="filter-btn" data-filter="your-category">
    <i class="fas fa-icon"></i> Category Name
</button>
```
2. Use `data-category="your-category"` in note cards

---

## General Tips

### Where to Find Icons
Use [Font Awesome](https://fontawesome.com/icons):
- Data: `fa-database`, `fa-chart-bar`, `fa-chart-line`
- AI/ML: `fa-brain`, `fa-robot`, `fa-network-wired`
- Code: `fa-code`, `fa-laptop-code`, `fa-terminal`
- Documents: `fa-file-pdf`, `fa-book`, `fa-file-alt`

### Color Scheme
Current colors (defined in CSS):
- Primary: `#6366f1` (Purple)
- Secondary: `#8b5cf6` (Light Purple)
- Accent: `#ec4899` (Pink)
- Success: `#10b981` (Green)

### Responsive Design
All sections auto-adjust for mobile:
- Featured carousel: Slides adapt to screen size
- Completed projects: 1 column on mobile
- Notes: 1 column on mobile
- All images scale properly

### Testing Locally
1. Open `index.html` in browser
2. Test carousel navigation
3. Test notes filtering
4. Test modal viewing
5. Check mobile view (F12 → Device toolbar)

### Pushing to GitHub
After making changes:
```bash
git add .
git commit -m "Updated projects and notes"
git push origin main
```

Or use GitHub Desktop (see GITHUB-DESKTOP-GUIDE.txt)

---

## Quick Reference

### Current Dummy Content:
- **Background**: 5 tech-themed images
- **Featured Projects**: 3 AI/ML projects (LLM, Sentiment, RAG)
- **Completed Projects**: 6 data science projects
- **Notes**: 6 example notes (3 external, 3 embedded)

### File Structure:
```
my-portfolio/
├── index.html          (Main content - UPDATE THIS)
├── style.css           (Styling - usually no changes needed)
├── script.js           (Functionality - usually no changes needed)
└── HOW-TO-UPDATE.md    (This guide)
```

### Common Tasks:
1. **Add slideshow image**: Line ~35, add `<div class="slide" style="..."></div>`
2. **Add featured project**: Line ~165, copy slide block
3. **Add completed project**: Line ~285, copy card block
4. **Add note**: Line ~445, copy note card
5. **Update progress**: Find project, change `width: XX%`

---

## Need Help?

1. Check HTML comments in code (marked with `<!-- -->`)
2. Each section has clear markers
3. Copy existing examples and modify
4. Test locally before pushing

## Resources

- [Unsplash](https://unsplash.com) - Free images
- [Font Awesome](https://fontawesome.com) - Icons
- [Google Drive](https://drive.google.com) - Host external notes
- [GitHub Pages](https://pages.github.com) - Live site

---

**Remember**: All features are designed to scale infinitely. Add as many items as you want - the design will adapt automatically!
