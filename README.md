# Nurul Labib Sayeedi — Personal Website

Personal website and academic portfolio of **Nurul Labib Sayeedi**, a 3rd-year B.Sc. Data Science student at United International University (UIU), Dhaka.

Live at: **[labib1610.github.io](https://labib1610.github.io)**

---

## Repository Structure

```
Labib1610.github.io/
│
├── index.html              # Landing page (projects, research, notes, contact)
├── style.css               # Styles for the landing page
├── script.js               # JS for the landing page
│
├── assets/
│   ├── profile.jpeg        # Profile photo
│   └── Labib_CV.pdf        # CV (place here — linked from academic portfolio)
│
├── portfolio-website/      # Academic portfolio (About, Publications, etc.)
│   ├── index.html          # About me
│   ├── publications.html   # 7 research papers (3 conference, 1 workshop, 3 preprints)
│   ├── projects.html       # Dev projects
│   ├── honor_achievement.html  # Scholarships
│   ├── blog/               # Blog posts + PDF notes
│   │   ├── index.html
│   │   ├── kalman-filter.html
│   │   ├── decision-models-from-scratch.html
│   │   ├── Kalman_Filter_Notes.pdf
│   │   └── Decision_Models_from_Scratch.pdf
│   ├── css/                # Bootstrap, Ionicons, custom CSS
│   ├── js/                 # jQuery, Bootstrap, pace, site script
│   ├── fonts/              # Ionicons web font files
│   └── img/
│       ├── icon.png        # Site favicon / logo
│       └── labib.jpeg      # Profile photo (copy of assets/profile.jpeg)
│
└── ADDING_CONTENT.md       # Guide for adding new papers and projects
```

---

## Pages

| URL | Description |
|-----|-------------|
| `labib1610.github.io` | Landing page — projects, research highlights, notes |
| `labib1610.github.io/portfolio-website/` | Academic portfolio — About |
| `labib1610.github.io/portfolio-website/publications.html` | 7 research papers |
| `labib1610.github.io/portfolio-website/projects.html` | Dev projects |
| `labib1610.github.io/portfolio-website/blog/` | Blog & PDF notes |
| `labib1610.github.io/portfolio-website/honor_achievement.html` | Awards & scholarships |

---

## Research

**Conference Papers**
- BanglaVerse — EMNLP Findings 2026
- BanglaRiddleEval — *SEM 2026
- Bangla Idioms — EACL Findings 2027

**Workshop Paper**
- MULTIHALUDET — MeLLM @ ACL 2026

**Preprints**
- P2NPCBench (arXiv)
- Mind the Scale (arXiv)
- Baby and Me (ResearchSquare)

---

## Updating the Site

- **Add a new paper or project:** follow the step-by-step guide in [`ADDING_CONTENT.md`](ADDING_CONTENT.md)
- **CV:** place your PDF at `assets/Labib_CV.pdf` — the academic portfolio already links to it
- **Profile photo:** replace `assets/profile.jpeg` and `portfolio-website/img/labib.jpeg`

To push changes:
```bash
git add .
git commit -m "Update: describe your changes"
git push
```

GitHub Pages will redeploy automatically within a few minutes.

---

Built with HTML, CSS, and JavaScript. Academic portfolio design by [@UIU Developers Hub](https://github.com/UIU-Developers-Hub).
