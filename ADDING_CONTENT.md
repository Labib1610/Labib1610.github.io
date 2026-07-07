# Adding Papers & Projects to the Portfolio

A step-by-step guide for adding new research papers and projects — completed or ongoing — without touching unrelated code.

---

## Table of Contents

1. [Add a Completed Research Paper](#1-add-a-completed-research-paper)
2. [Add a Completed Non-Research Project](#2-add-a-completed-non-research-project)
3. [Add an Ongoing Project (In Progress)](#3-add-an-ongoing-project-in-progress)
4. [Update an Existing Ongoing Project to Completed](#4-update-an-existing-ongoing-project-to-completed)
5. [Update the "Read More" Abstract](#5-update-the-read-more-abstract)
6. [Quick Reference: Status Badges](#6-quick-reference-status-badges)

---

## 1. Add a Completed Research Paper

### Step A — Add the card in `index.html`

Open `index.html` and find the **Completed Projects** section:

```html
<!-- Completed Projects Section -->
<section id="completed-projects" ...>
    ...
    <div class="completed-projects-grid">
```

Paste a new card block **inside** the `completed-projects-grid` div, before the existing cards (newest first):

```html
<!-- Research Paper - YOUR TITLE -->
<div class="completed-project-card research-card">
    <div class="project-header">
        <div class="project-icon-wrapper">
            <i class="fas fa-brain"></i>   <!-- pick any Font Awesome icon -->
        </div>
        <span class="completion-badge"><i class="fas fa-check-circle"></i> Accepted</span>
    </div>
    <span class="venue-badge"><i class="fas fa-award"></i> VENUE_NAME · YEAR</span>
    <h3>Full Paper Title Here</h3>
    <p>One or two sentences summarising the contribution and key result.</p>
    <div class="project-tech">
        <span class="tech-tag">Python</span>
        <span class="tech-tag">LLMs</span>
        <!-- add more tags as needed -->
    </div>
    <div class="project-links">
        <a href="PAPER_URL" class="project-link" target="_blank">
            <i class="fas fa-file-pdf"></i> Paper
        </a>
        <a href="GITHUB_URL" class="project-link" target="_blank">
            <i class="fab fa-github"></i> GitHub
        </a>
        <!-- optional: dataset, website, demo links follow the same pattern -->
        <button class="project-link read-more-btn" data-paper-id="YOUR_ID">
            <i class="fas fa-book-open"></i> Read More
        </button>
    </div>
</div>
```

**Rules:**
- `data-paper-id` must be a unique lowercase string with no spaces (e.g. `"mypaperv2"`). You will use this same string in Step B.
- If the paper is **not yet accepted**, change the badge to:
  ```html
  <span class="completion-badge preprint-badge"><i class="fas fa-clock"></i> Under Review</span>
  ```
  and remove the `<span class="venue-badge">` line entirely (or add it later when accepted).

---

### Step B — Register the abstract in `script.js`

Open `script.js` and find the `paperAbstracts` object (search for `const paperAbstracts`). Add a new entry:

```javascript
const paperAbstracts = {
    // ... existing entries ...

    your_id: {
        title: "Full Paper Title Here",
        venue: "VENUE_NAME · YEAR",   // or "Under Review"
        abstract: "Paste the full abstract text here as a single string.",
        links: [
            { icon: "fas fa-file-pdf", label: "Paper",  url: "PAPER_URL" },
            { icon: "fab fa-github",   label: "GitHub", url: "GITHUB_URL" },
            // add more as needed: dataset, website, demo
        ]
    }
};
```

The `your_id` key must exactly match the `data-paper-id` you set in Step A.

---

## 2. Add a Completed Non-Research Project

Open `index.html`, find the `completed-projects-grid` div, and paste a simpler card (no venue badge, no Read More):

```html
<!-- Project - YOUR PROJECT NAME -->
<div class="completed-project-card">
    <div class="project-header">
        <div class="project-icon-wrapper">
            <i class="fas fa-code"></i>   <!-- pick any icon -->
        </div>
        <span class="completion-badge"><i class="fas fa-check-circle"></i> Completed</span>
    </div>
    <h3>Project Name</h3>
    <p>Brief description of what the project does and what makes it interesting.</p>
    <div class="project-tech">
        <span class="tech-tag">JavaScript</span>
        <span class="tech-tag">HTML</span>
        <span class="tech-tag">CSS</span>
    </div>
    <div class="project-links">
        <a href="GITHUB_URL" class="project-link" target="_blank">
            <i class="fab fa-github"></i> GitHub
        </a>
        <a href="DEMO_URL" class="project-link" target="_blank">
            <i class="fas fa-external-link-alt"></i> Live Demo
        </a>
    </div>
</div>
```

No changes to `script.js` needed for non-research projects without abstracts.

---

## 3. Add an Ongoing Project (In Progress)

Ongoing projects appear in the **Featured Projects** carousel at the top of the page.

### Step A — Add a new slide in `index.html`

Find the `carousel-track` div inside `featured-projects-section` and add a new slide. Make the **first** (default) slide have `class="featured-project-slide active"` and all others just `class="featured-project-slide"`:

```html
<!-- Ongoing Project - YOUR TITLE -->
<div class="featured-project-slide">
    <div class="featured-project-card">
        <div class="featured-project-image">
            <img src="IMAGE_URL" alt="Project Image">
            <div class="project-status">
                <span class="status-badge working">
                    <i class="fas fa-spinner fa-spin"></i> In Progress
                </span>
            </div>
        </div>
        <div class="featured-project-content">
            <h3>Your Project Title</h3>
            <p>Short description of what you are building and why it matters.</p>
            <div class="project-tech">
                <span class="tech-tag">Python</span>
                <!-- add more tags -->
            </div>
            <div class="project-progress">
                <div class="progress-bar">
                    <div class="progress-fill" style="width: 50%"></div>
                    <!-- change 50% to your actual completion percentage -->
                </div>
                <span class="progress-text">50% Complete</span>
            </div>
        </div>
    </div>
</div>
```

### Step B — Add a dot for the new slide

Find the `carousel-dots` div and add one more dot:

```html
<div class="carousel-dots" id="featuredDots">
    <span class="dot active" data-slide="0"></span>
    <span class="dot" data-slide="1"></span>
    <span class="dot" data-slide="2"></span>  <!-- new dot -->
</div>
```

The `data-slide` index must match the slide's position (0-based).

**No changes to `script.js` needed** — the carousel picks up new slides automatically.

---

## 4. Update an Existing Ongoing Project to Completed

When a featured project finishes and gets accepted:

1. **Remove the slide** from the `carousel-track` in `index.html`.
2. **Remove the corresponding dot** from `carousel-dots` and re-number the remaining dots starting from 0.
3. **Add a completed card** to `completed-projects-grid` following Section 1 or 2 above.
4. If it has an abstract, **add the entry** to `paperAbstracts` in `script.js`.

---

## 5. Update the "Read More" Abstract

When a paper's venue or abstract changes after submission:

- Open `script.js`.
- Find the paper's entry in the `paperAbstracts` object by its `id`.
- Update the `venue` or `abstract` fields directly.

To add a venue to a paper that was previously "Under Review":

1. In `index.html`, change the badge:
   ```html
   <!-- Before -->
   <span class="completion-badge preprint-badge"><i class="fas fa-clock"></i> Under Review</span>

   <!-- After -->
   <span class="completion-badge"><i class="fas fa-check-circle"></i> Accepted</span>
   ```
2. Add the venue badge line below the project-header div:
   ```html
   <span class="venue-badge"><i class="fas fa-award"></i> CONFERENCE · YEAR</span>
   ```
3. In `script.js`, update the `venue` field for that paper's entry in `paperAbstracts`.

---

## 6. Quick Reference: Status Badges

| Situation | Badge HTML |
|---|---|
| Accepted paper | `<span class="completion-badge"><i class="fas fa-check-circle"></i> Accepted</span>` |
| Under review / preprint | `<span class="completion-badge preprint-badge"><i class="fas fa-clock"></i> Under Review</span>` |
| Finished non-research project | `<span class="completion-badge"><i class="fas fa-check-circle"></i> Completed</span>` |
| In-progress project | `<span class="status-badge working"><i class="fas fa-spinner fa-spin"></i> In Progress</span>` |

---

## Font Awesome Icon Suggestions

| Research topic | Icon class |
|---|---|
| NLP / Language | `fas fa-language` |
| Vision / Multimodal | `fas fa-eye` |
| Hallucination / Safety | `fas fa-search` |
| Reasoning / Benchmarks | `fas fa-brain` |
| Graphs / Geometry | `fas fa-project-diagram` |
| Data / Datasets | `fas fa-database` |
| General ML | `fas fa-robot` |
| Web / Frontend | `fas fa-code` |
| Game | `fas fa-gamepad` |

Browse more icons at [fontawesome.com/icons](https://fontawesome.com/icons).
