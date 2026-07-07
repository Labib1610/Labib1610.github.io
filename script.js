/* ========================================
   Portfolio Script
   ======================================== */

// === Typewriter Animation ===
const roles = [
    "Researcher",
    "Data Scientist",
    "NLP Engineer",
    "ML Engineer"
];

let roleIndex = 0;
let charIndex = 0;
let isDeleting = false;
const typewriterEl = document.getElementById("typewriter");

function typeWriter() {
    if (!typewriterEl) return;
    const current = roles[roleIndex];
    if (isDeleting) {
        typewriterEl.textContent = current.substring(0, charIndex - 1);
        charIndex--;
    } else {
        typewriterEl.textContent = current.substring(0, charIndex + 1);
        charIndex++;
    }
    let speed = isDeleting ? 60 : 100;
    if (!isDeleting && charIndex === current.length) {
        speed = 1800;
        isDeleting = true;
    } else if (isDeleting && charIndex === 0) {
        isDeleting = false;
        roleIndex = (roleIndex + 1) % roles.length;
        speed = 300;
    }
    setTimeout(typeWriter, speed);
}
typeWriter();

// === Navbar Mobile Toggle ===
const navToggle = document.getElementById("navToggle");
const navLinks = document.getElementById("navLinks");

if (navToggle && navLinks) {
    navToggle.addEventListener("click", () => {
        navLinks.classList.toggle("active");
        navToggle.classList.toggle("active");
    });
    document.querySelectorAll(".nav-link").forEach(link => {
        link.addEventListener("click", () => {
            navLinks.classList.remove("active");
            navToggle.classList.remove("active");
        });
    });
}

// Navbar scroll effect
window.addEventListener("scroll", () => {
    const navbar = document.querySelector(".navbar");
    if (navbar) {
        navbar.classList.toggle("scrolled", window.scrollY > 50);
    }
    const scrollTopBtn = document.getElementById("scrollTop");
    if (scrollTopBtn) {
        scrollTopBtn.classList.toggle("visible", window.scrollY > 300);
    }
    updateActiveNavLink();
});

function updateActiveNavLink() {
    const sections = document.querySelectorAll("section[id]");
    const scrollPos = window.scrollY + 100;
    sections.forEach(section => {
        const top = section.offsetTop;
        const height = section.offsetHeight;
        const id = section.getAttribute("id");
        const link = document.querySelector(`.nav-link[href="#${id}"]`);
        if (link) {
            link.classList.toggle("active", scrollPos >= top && scrollPos < top + height);
        }
    });
}

// === Scroll to Top ===
const scrollTopBtn = document.getElementById("scrollTop");
if (scrollTopBtn) {
    scrollTopBtn.addEventListener("click", () => {
        window.scrollTo({ top: 0, behavior: "smooth" });
    });
}

// === Featured Projects Carousel ===
const slides = document.querySelectorAll(".featured-project-slide");
const dots = document.querySelectorAll("#featuredDots .dot");
let currentSlide = 0;
let autoSlideTimer;

const carouselTrack = document.querySelector(".carousel-track");

function goToSlide(index) {
    if (!slides.length) return;
    dots[currentSlide]?.classList.remove("active");
    currentSlide = (index + slides.length) % slides.length;
    dots[currentSlide]?.classList.add("active");
    if (carouselTrack) {
        carouselTrack.style.transform = `translateX(-${currentSlide * 100}%)`;
    }
}

function startAutoSlide() {
    autoSlideTimer = setInterval(() => goToSlide(currentSlide + 1), 5000);
}

function resetAutoSlide() {
    clearInterval(autoSlideTimer);
    startAutoSlide();
}

document.getElementById("prevFeatured")?.addEventListener("click", () => {
    goToSlide(currentSlide - 1);
    resetAutoSlide();
});

document.getElementById("nextFeatured")?.addEventListener("click", () => {
    goToSlide(currentSlide + 1);
    resetAutoSlide();
});

dots.forEach((dot, i) => {
    dot.addEventListener("click", () => {
        goToSlide(i);
        resetAutoSlide();
    });
});

if (slides.length > 1) startAutoSlide();

// === Notes Filter ===
const filterBtns = document.querySelectorAll(".filter-btn");
const noteCards = document.querySelectorAll(".note-card");

filterBtns.forEach(btn => {
    btn.addEventListener("click", () => {
        filterBtns.forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        const category = btn.dataset.category;
        noteCards.forEach(card => {
            const show = category === "all" || card.dataset.category === category;
            card.style.display = show ? "" : "none";
        });
    });
});

// === Note Modal ===
const noteModal = document.getElementById("noteModal");
const closeNoteModal = document.getElementById("closeNoteModal");

document.querySelectorAll(".view-note-btn").forEach(btn => {
    btn.addEventListener("click", () => {
        if (noteModal) {
            document.getElementById("noteModalBody").innerHTML = "<p>Loading...</p>";
            noteModal.classList.add("active");
        }
    });
});

closeNoteModal?.addEventListener("click", () => noteModal?.classList.remove("active"));
noteModal?.addEventListener("click", e => {
    if (e.target === noteModal) noteModal.classList.remove("active");
});

// === Paper Abstract Modal ===
const paperAbstracts = {
    multihaludet: {
        title: "MULTIHALUDET: Multilingual Hallucination Detection via LLM Hidden State Probing",
        venue: "MeLLM Workshop · ACL 2026",
        abstract: "Hallucinations in Large Language Models (LLMs) represent a critical barrier to their reliable deployment, a vulnerability heavily exacerbated in non-English and resource-constrained contexts. Existing detection approaches that rely on output confidence heuristics or single-layer internal representations frequently fail to capture deep, complex factual inconsistencies across diverse languages. To address this, we introduce MultiHaluDet, a novel three-stage stacking framework that detects multilingual hallucinations by probing the full hidden state trajectories of frozen LLMs without requiring language-specific fine-tuning. Our method extracts sequential features across multiple layers and processes them via a hybrid architecture using multi-scale attention and self-attention pooling. By generating out-of-fold embeddings that feed into a calibrated classical classifier ensemble, MultiHaluDet captures both fine-grained and coarse-grained patterns of factual inconsistency. Extensive experiments demonstrate that our framework achieves state-of-the-art detection performance, reaching up to 98.55% AUROC on the English HaluEval and TriviaQA benchmarks using Mistral-7B and LLaMA2-7B architectures. Crucially, we rigorously evaluate our framework's cross-lingual generalization across high (French), medium (Bangla), and low-resource (Amharic) languages. MultiHaluDet demonstrates exceptional representational robustness, consistently outperforming baselines and successfully transferring hallucination detection capabilities across typologically diverse linguistic tiers.",
        links: [
            { icon: "fas fa-file-pdf", label: "Paper", url: "https://aclanthology.org/2026.mellm-1.6.pdf" },
            { icon: "fab fa-github", label: "GitHub", url: "https://github.com/alvi-uiu/MULTIHALUDET" }
        ]
    },
    banglariddle: {
        title: "Can LLMs Solve My Grandma's Riddle? Evaluating Multilingual Large Language Models on Reasoning Traditional Bangla Tricky Riddles",
        venue: "*SEM 2026",
        abstract: "Large Language Models (LLMs) show impressive performance on many NLP benchmarks, yet their ability to reason in figurative, culturally grounded, and low-resource settings remains underexplored. We address this gap for Bangla by introducing BanglaRiddleEval, a benchmark of 1,244 traditional Bangla riddles instantiated across four tasks (4,976 riddle-task artifacts in total). Using an LLM-based pipeline, we generate Chain-of-Thought explanations, semantically coherent distractors, and fine-grained ambiguity annotations, and evaluate a diverse suite of open-source and closed-source models under different prompting strategies. Models achieve moderate semantic overlap on generative QA but low correctness, MCQ accuracy peaks at only about 56% versus an 83.3% human baseline, and ambiguity resolution ranges from roughly 26% to 68%, with high-quality explanations confined to the strongest models. These results show that current LLMs capture some cues needed for Bangla riddle reasoning but remain far from human-level performance, establishing BanglaRiddleEval as a challenging new benchmark for low-resource figurative reasoning. All data, code, and evaluation scripts are available on GitHub.",
        links: [
            { icon: "fas fa-file-pdf", label: "Paper", url: "https://aclanthology.org/2026.starsem-conference.22.pdf" },
            { icon: "fab fa-github", label: "GitHub", url: "https://github.com/Labib1610/BanglaRiddleEval" }
        ]
    },
    banglaverse: {
        title: "Many Dialects, Many Languages, One Cultural Lens: Evaluating Multilingual VLMs for Bengali Culture Understanding Across Historically Linked Languages and Regional Dialects",
        venue: "Under Review",
        abstract: "Bangla culture is richly expressed through region, dialect, history, food, politics, media, and everyday visual life, yet it remains underrepresented in multimodal evaluation. To address this gap, we introduce BanglaVerse, a culturally grounded benchmark for evaluating multilingual vision–language models (VLMs) on Bengali culture across historically linked languages and regional dialects. Built from 1,152 manually curated images across nine domains, the benchmark supports visual question answering and captioning, and is expanded into four languages and five Bangla dialects, yielding ∼32.3K artifacts. Our experiments show that evaluating only standard Bangla overestimates true model capability: performance drops under dialectal variation, especially for caption generation, while historically linked languages such as Hindi and Urdu retain some cultural meaning but remain weaker for structured reasoning. Across domains, the main bottleneck is missing cultural knowledge rather than visual grounding alone, with knowledge-intensive categories. These findings position BanglaVerse as a more realistic test bed for measuring culturally grounded multimodal understanding under linguistic variation.",
        links: [
            { icon: "fas fa-file-pdf", label: "Paper", url: "https://arxiv.org/pdf/2603.21165" },
            { icon: "fab fa-github", label: "GitHub", url: "https://github.com/faiyazabdullah/BanglaVerse" },
            { icon: "fas fa-database", label: "Dataset", url: "https://huggingface.co/datasets/FaiyazAbdullah114708/BanglaVerse" },
            { icon: "fas fa-external-link-alt", label: "Website", url: "https://labib1610.github.io/BanglaVerse/" }
        ]
    }
};

const paperModal = document.getElementById("paperModal");
const paperModalBody = document.getElementById("paperModalBody");
const closePaperModal = document.getElementById("closePaperModal");

document.querySelectorAll(".read-more-btn").forEach(btn => {
    btn.addEventListener("click", () => {
        const id = btn.dataset.paperId;
        const paper = paperAbstracts[id];
        if (!paper || !paperModal) return;

        const linksHtml = paper.links.map(l =>
            `<a href="${l.url}" target="_blank"><i class="${l.icon}"></i> ${l.label}</a>`
        ).join("");

        paperModalBody.innerHTML = `
            <h2 class="paper-abstract-title">${paper.title}</h2>
            <span class="paper-abstract-venue"><i class="fas fa-award"></i> ${paper.venue}</span>
            <p class="paper-abstract-text">${paper.abstract}</p>
            <div class="paper-abstract-links">${linksHtml}</div>
        `;
        paperModal.classList.add("active");
    });
});

closePaperModal?.addEventListener("click", () => paperModal?.classList.remove("active"));
paperModal?.addEventListener("click", e => {
    if (e.target === paperModal) paperModal.classList.remove("active");
});

// Close modals on Escape key
document.addEventListener("keydown", e => {
    if (e.key === "Escape") {
        noteModal?.classList.remove("active");
        paperModal?.classList.remove("active");
    }
});

// === Scroll Reveal Animation ===
const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.classList.add("visible");
        }
    });
}, { threshold: 0.1 });

document.querySelectorAll(
    ".completed-project-card, .skill-category, .note-card, .about-text"
).forEach(el => observer.observe(el));
