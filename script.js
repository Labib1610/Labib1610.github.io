// =======================================
// Professional Portfolio JavaScript
// =======================================

// === Typing Animation ===
const textElement = document.getElementById('typewriter');
const roles = [
    "Data Scientist",
    "Machine Learning Engineer",
    "AI Enthusiast",
    "LLM Developer"
];

let roleIndex = 0;
let charIndex = 0;
let isDeleting = false;

function type() {
    if (!textElement) return;
    
    const currentRole = roles[roleIndex];
    
    if (isDeleting) {
        textElement.textContent = currentRole.substring(0, charIndex - 1);
        charIndex--;
    } else {
        textElement.textContent = currentRole.substring(0, charIndex + 1);
        charIndex++;
    }

    if (!isDeleting && charIndex === currentRole.length) {
        isDeleting = true;
        setTimeout(type, 2000);
    } else if (isDeleting && charIndex === 0) {
        isDeleting = false;
        roleIndex = (roleIndex + 1) % roles.length;
        setTimeout(type, 500);
    } else {
        setTimeout(type, isDeleting ? 50 : 150);
    }
}

// === Smooth Scrolling for Navigation Links ===
function setupSmoothScrolling() {
    const navLinks = document.querySelectorAll('.nav-link');
    
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href');
            const targetSection = document.querySelector(targetId);
            
            if (targetSection) {
                const navHeight = document.querySelector('.navbar').offsetHeight;
                const targetPosition = targetSection.offsetTop - navHeight;
                
                window.scrollTo({
                    top: targetPosition,
                    behavior: 'smooth'
                });
                
                // Close mobile menu if open
                const navLinksContainer = document.getElementById('navLinks');
                if (navLinksContainer) {
                    navLinksContainer.classList.remove('active');
                }
                
                // Update active link
                navLinks.forEach(l => l.classList.remove('active'));
                this.classList.add('active');
            }
        });
    });
}

// === Mobile Navigation Toggle ===
function setupMobileNav() {
    const navToggle = document.getElementById('navToggle');
    const navLinks = document.getElementById('navLinks');
    
    if (navToggle && navLinks) {
        navToggle.addEventListener('click', () => {
            navLinks.classList.toggle('active');
            navToggle.classList.toggle('active');
        });
        
        // Close menu when clicking outside
        document.addEventListener('click', (e) => {
            if (!navToggle.contains(e.target) && !navLinks.contains(e.target)) {
                navLinks.classList.remove('active');
                navToggle.classList.remove('active');
            }
        });
    }
}

// === Navbar Scroll Effect ===
function setupNavbarScroll() {
    const navbar = document.querySelector('.navbar');
    let lastScroll = 0;
    
    window.addEventListener('scroll', () => {
        const currentScroll = window.pageYOffset;
        
        if (currentScroll > 100) {
            navbar.classList.add('scrolled');
        } else {
            navbar.classList.remove('scrolled');
        }
        
        lastScroll = currentScroll;
    });
}

// === Active Section Highlighting ===
function setupActiveSection() {
    const sections = document.querySelectorAll('section[id]');
    const navLinks = document.querySelectorAll('.nav-link');
    
    window.addEventListener('scroll', () => {
        let current = '';
        const scrollPosition = window.pageYOffset + 200;
        
        sections.forEach(section => {
            const sectionTop = section.offsetTop;
            const sectionHeight = section.offsetHeight;
            
            if (scrollPosition >= sectionTop && scrollPosition < sectionTop + sectionHeight) {
                current = section.getAttribute('id');
            }
        });
        
        navLinks.forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('href') === `#${current}`) {
                link.classList.add('active');
            }
        });
    });
}

// === Scroll to Top Button ===
function setupScrollToTop() {
    const scrollTopBtn = document.getElementById('scrollTop');
    
    if (scrollTopBtn) {
        window.addEventListener('scroll', () => {
            if (window.pageYOffset > 500) {
                scrollTopBtn.classList.add('visible');
            } else {
                scrollTopBtn.classList.remove('visible');
            }
        });
        
        scrollTopBtn.addEventListener('click', () => {
            window.scrollTo({
                top: 0,
                behavior: 'smooth'
            });
        });
    }
}

// === Scroll Animations (Intersection Observer) ===
function setupScrollAnimations() {
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, observerOptions);
    
    // Animate cards
    const animateElements = document.querySelectorAll('.project-card, .skill-category, .stat-item, .contact-item');
    animateElements.forEach(element => {
        element.style.opacity = '0';
        element.style.transform = 'translateY(30px)';
        element.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(element);
    });
}

// === Cursor Effect (Optional - Desktop Only) ===
function setupCursorEffect() {
    if (window.innerWidth > 768) {
        const cursor = document.createElement('div');
        cursor.classList.add('custom-cursor');
        cursor.style.cssText = `
            position: fixed;
            width: 20px;
            height: 20px;
            border: 2px solid #6366f1;
            border-radius: 50%;
            pointer-events: none;
            z-index: 9999;
            transition: transform 0.2s ease;
            display: none;
        `;
        document.body.appendChild(cursor);
        
        document.addEventListener('mousemove', (e) => {
            cursor.style.display = 'block';
            cursor.style.left = e.clientX + 'px';
            cursor.style.top = e.clientY + 'px';
        });
        
        // Scale cursor on interactive elements
        const interactiveElements = document.querySelectorAll('a, button, .project-card, .skill-category');
        interactiveElements.forEach(element => {
            element.addEventListener('mouseenter', () => {
                cursor.style.transform = 'scale(1.5)';
                cursor.style.borderColor = '#ec4899';
            });
            element.addEventListener('mouseleave', () => {
                cursor.style.transform = 'scale(1)';
                cursor.style.borderColor = '#6366f1';
            });
        });
    }
}

// === Update Copyright Year ===
function updateCopyrightYear() {
    const footer = document.querySelector('.footer p');
    if (footer) {
        const year = new Date().getFullYear();
        footer.innerHTML = footer.innerHTML.replace('2025', year);
    }
}

// === Preload Animation ===
function setupPreloader() {
    window.addEventListener('load', () => {
        const preloader = document.querySelector('.preloader');
        if (preloader) {
            preloader.style.opacity = '0';
            setTimeout(() => {
                preloader.style.display = 'none';
            }, 500);
        }
    });
}

// === Initialize All Functions ===
document.addEventListener('DOMContentLoaded', () => {
    // Start typing animation
    type();
    
    // Setup all features
    setupSmoothScrolling();
    setupMobileNav();
    setupNavbarScroll();
    setupActiveSection();
    setupScrollToTop();
    setupScrollAnimations();
    setupCursorEffect();
    updateCopyrightYear();
    setupPreloader();
    
    console.log('🚀 Portfolio loaded successfully!');
});

// === Prevent FOUC (Flash of Unstyled Content) ===
document.documentElement.classList.add('js-enabled');

// === Performance Optimization ===
let resizeTimer;
window.addEventListener('resize', () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
        // Recalculate positions if needed
        console.log('Window resized');
    }, 250);
});
// =======================================
// Background Slideshow
// =======================================
function initBackgroundSlideshow() {
    const slides = document.querySelectorAll('.bg-slideshow .slide');
    let currentSlide = 0;
    
    function showNextSlide() {
        slides[currentSlide].classList.remove('active');
        currentSlide = (currentSlide + 1) % slides.length;
        slides[currentSlide].classList.add('active');
    }
    
    // Change slide every 6 seconds
    setInterval(showNextSlide, 6000);
}

// =======================================
// Featured Projects Carousel
// =======================================
function initFeaturedCarousel() {
    const carouselTrack = document.querySelector('.carousel-track');
    const slides = document.querySelectorAll('.featured-project-slide');
    const prevBtn = document.querySelector('.prev-btn');
    const nextBtn = document.querySelector('.next-btn');
    const dots = document.querySelectorAll('.carousel-dots .dot');
    
    if (!carouselTrack || slides.length === 0) return;
    
    let currentIndex = 0;
    const slideWidth = slides[0].offsetWidth;
    
    function updateCarousel() {
        carouselTrack.style.transform = `translateX(-${currentIndex * 100}%)`;
        
        // Update active dot
        dots.forEach((dot, index) => {
            dot.classList.toggle('active', index === currentIndex);
        });
    }
    
    function nextSlide() {
        currentIndex = (currentIndex + 1) % slides.length;
        updateCarousel();
    }
    
    function prevSlide() {
        currentIndex = (currentIndex - 1 + slides.length) % slides.length;
        updateCarousel();
    }
    
    // Button listeners
    if (nextBtn) nextBtn.addEventListener('click', nextSlide);
    if (prevBtn) prevBtn.addEventListener('click', prevSlide);
    
    // Dot listeners
    dots.forEach((dot, index) => {
        dot.addEventListener('click', () => {
            currentIndex = index;
            updateCarousel();
        });
    });
    
    // Auto-advance (optional)
    setInterval(nextSlide, 8000);
    
    // Update on window resize
    window.addEventListener('resize', () => {
        updateCarousel();
    });
}

// =======================================
// Notes Filtering System
// =======================================
function initNotesFilter() {
    const filterButtons = document.querySelectorAll('.filter-btn');
    const noteCards = document.querySelectorAll('.note-card');
    
    if (!filterButtons.length) return;
    
    filterButtons.forEach(button => {
        button.addEventListener('click', () => {
            const filterValue = button.getAttribute('data-filter');
            
            // Update active button
            filterButtons.forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');
            
            // Filter cards
            noteCards.forEach(card => {
                const category = card.getAttribute('data-category');
                
                if (filterValue === 'all' || category === filterValue) {
                    card.style.display = 'block';
                    setTimeout(() => {
                        card.style.opacity = '1';
                        card.style.transform = 'translateY(0)';
                    }, 10);
                } else {
                    card.style.opacity = '0';
                    card.style.transform = 'translateY(20px)';
                    setTimeout(() => {
                        card.style.display = 'none';
                    }, 300);
                }
            });
        });
    });
}

// =======================================
// Note Modal System
// =======================================
function initNoteModal() {
    const modal = document.getElementById('noteModal');
    const modalBody = document.getElementById('modalBody');
    const modalClose = document.querySelector('.modal-close');
    const viewNoteBtns = document.querySelectorAll('.view-note-btn');
    
    if (!modal) return;
    
    // Open modal
    viewNoteBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const noteCard = btn.closest('.note-card');
            const noteTitle = noteCard.querySelector('h3').textContent;
            const noteContent = noteCard.getAttribute('data-content') || `
                <h2>${noteTitle}</h2>
                <h3>Introduction</h3>
                <p>This is a placeholder for your embedded note content. You can add full articles, tutorials, or detailed notes here.</p>
                
                <h3>Key Points</h3>
                <p>• Point 1: Detailed explanation</p>
                <p>• Point 2: More information</p>
                <p>• Point 3: Additional insights</p>
                
                <h3>Code Example</h3>
                <pre><code>
import torch
import torch.nn as nn

# Example neural network
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 5)
    
    def forward(self, x):
        return self.layer(x)
                </code></pre>
                
                <h3>Conclusion</h3>
                <p>Summary of the key learnings and takeaways from this note.</p>
            `;
            
            modalBody.innerHTML = noteContent;
            modal.classList.add('active');
            document.body.style.overflow = 'hidden';
        });
    });
    
    // Close modal
    function closeModal() {
        modal.classList.remove('active');
        document.body.style.overflow = '';
    }
    
    if (modalClose) {
        modalClose.addEventListener('click', closeModal);
    }
    
    // Close on background click
    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            closeModal();
        }
    });
    
    // Close on ESC key
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && modal.classList.contains('active')) {
            closeModal();
        }
    });
}

// =======================================
// Load More Projects
// =======================================
function initLoadMore() {
    const loadMoreBtn = document.getElementById('loadMoreProjects');
    
    if (loadMoreBtn) {
        loadMoreBtn.addEventListener('click', () => {
            // Add your logic to load more projects dynamically
            // This is a placeholder - you can fetch from API or load hidden elements
            alert('Load more functionality - Add your projects data source here!');
        });
    }
}

// =======================================
// Add New Note Button
// =======================================
function initAddNote() {
    const addNoteBtn = document.getElementById('addNoteBtn');
    
    if (addNoteBtn) {
        addNoteBtn.addEventListener('click', () => {
            // You can open a form or redirect to an admin panel
            alert('Add note functionality - Connect to your content management system!');
        });
    }
}

// =======================================
// Initialize All Functions on Load
// =======================================
document.addEventListener('DOMContentLoaded', () => {
    // Existing functions
    type();
    setupSmoothScrolling();
    setupMobileNav();
    setupNavbarScroll();
    setupIntersectionObserver();
    setupSkillBars();
    setupProjectFilters();
    setupFormValidation();
    
    // New functions
    initBackgroundSlideshow();
    initFeaturedCarousel();
    initNotesFilter();
    initNoteModal();
    initLoadMore();
    initAddNote();
});

// Handle page visibility for animations
document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        // Pause animations when tab is not visible
    } else {
        // Resume animations
    }
});
