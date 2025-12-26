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
    const prevBtn = document.getElementById('prevFeatured');
    const nextBtn = document.getElementById('nextFeatured');
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
            const slideIndex = parseInt(dot.getAttribute('data-slide'));
            currentIndex = slideIndex;
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
            const filterValue = button.getAttribute('data-category');
            
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
    const modalBody = document.getElementById('noteModalBody');
    const modalClose = document.getElementById('closeNoteModal');
    const viewNoteBtns = document.querySelectorAll('.view-note-btn');
    
    if (!modal) return;
    
    // Predefined note contents
    const noteContents = {
        'note-1': `
            <h2>Neural Networks Fundamentals</h2>
            <h3>Introduction to Neural Networks</h3>
            <p>Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) organized in layers that process information through weighted connections.</p>
            
            <h3>Key Components</h3>
            <p><strong>Input Layer:</strong> Receives raw data features</p>
            <p><strong>Hidden Layers:</strong> Process and transform data through activation functions</p>
            <p><strong>Output Layer:</strong> Produces final predictions or classifications</p>
            
            <h3>Activation Functions</h3>
            <p>Common activation functions include:</p>
            <ul>
                <li><strong>ReLU:</strong> max(0, x) - Most popular for hidden layers</li>
                <li><strong>Sigmoid:</strong> 1/(1+e^-x) - Used for binary classification</li>
                <li><strong>Softmax:</strong> Used for multi-class classification</li>
            </ul>
            
            <h3>Code Example: Simple Neural Network</h3>
            <pre><code>import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Create model instance
model = NeuralNetwork(784, 128, 10)
print(model)</code></pre>
            
            <h3>Backpropagation</h3>
            <p>Backpropagation is the key algorithm for training neural networks. It calculates gradients of the loss function with respect to the weights using the chain rule of calculus.</p>
            
            <h3>Training Process</h3>
            <p>1. Forward pass: Input data flows through network<br>
            2. Calculate loss: Compare predictions with actual labels<br>
            3. Backward pass: Compute gradients using backpropagation<br>
            4. Update weights: Apply gradient descent to minimize loss</p>
            
            <h3>Conclusion</h3>
            <p>Understanding these fundamentals is crucial for building effective deep learning models. Practice implementing networks from scratch to solidify these concepts.</p>
        `,
        'note-2': `
            <h2>PyTorch Tutorial Series</h2>
            <h3>Getting Started with PyTorch</h3>
            <p>PyTorch is an open-source machine learning library developed by Facebook's AI Research lab. It provides tensor computation with GPU acceleration and automatic differentiation.</p>
            
            <h3>Installation</h3>
            <pre><code>pip install torch torchvision torchaudio</code></pre>
            
            <h3>Basic Tensor Operations</h3>
            <pre><code>import torch

# Create tensors
x = torch.tensor([1, 2, 3])
y = torch.randn(3, 3)

# Operations
z = x + 5
matrix_mult = torch.matmul(y, y.T)

# GPU acceleration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = x.to(device)</code></pre>
            
            <h3>Building a Simple Model</h3>
            <pre><code>import torch.nn as nn
import torch.optim as optim

# Define model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 50)
        self.linear2 = nn.Linear(50, 1)
        
    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = self.linear2(x)
        return x

model = SimpleModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)</code></pre>
            
            <h3>Training Loop</h3>
            <pre><code>for epoch in range(100):
    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')</code></pre>
            
            <h3>Key Concepts</h3>
            <p>• <strong>Autograd:</strong> Automatic differentiation engine<br>
            • <strong>DataLoader:</strong> Efficient data loading and batching<br>
            • <strong>nn.Module:</strong> Base class for all neural network modules<br>
            • <strong>Optimizers:</strong> Implement optimization algorithms</p>
            
            <h3>Next Steps</h3>
            <p>Practice building CNNs for image classification, RNNs for sequence data, and transformers for NLP tasks. Experiment with different architectures and hyperparameters.</p>
        `,
        'note-3': `
            <h2>CNN Architectures Comparison</h2>
            <h3>Overview</h3>
            <p>Convolutional Neural Networks (CNNs) have revolutionized computer vision. This note compares popular architectures and their key innovations.</p>
            
            <h3>LeNet-5 (1998)</h3>
            <p><strong>Layers:</strong> 7 (Conv-Pool-Conv-Pool-FC-FC-Output)<br>
            <strong>Innovation:</strong> First successful CNN for handwritten digits<br>
            <strong>Parameters:</strong> ~60K</p>
            
            <h3>AlexNet (2012)</h3>
            <p><strong>Innovation:</strong> Deep CNN with ReLU, dropout, and data augmentation<br>
            <strong>Parameters:</strong> ~60M<br>
            <strong>Impact:</strong> Won ImageNet 2012, sparked deep learning revolution</p>
            
            <h3>VGG-16 (2014)</h3>
            <p><strong>Key Idea:</strong> Use small 3x3 filters throughout<br>
            <strong>Depth:</strong> 16-19 layers<br>
            <strong>Parameters:</strong> ~138M<br>
            <strong>Advantage:</strong> Simple, uniform architecture</p>
            <pre><code># VGG Block Example
import torch.nn as nn

class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_convs):
        super().__init__()
        layers = []
        for _ in range(num_convs):
            layers.append(nn.Conv2d(in_channels, out_channels, 3, padding=1))
            layers.append(nn.ReLU())
            in_channels = out_channels
        layers.append(nn.MaxPool2d(2, 2))
        self.block = nn.Sequential(*layers)</code></pre>
            
            <h3>ResNet (2015)</h3>
            <p><strong>Innovation:</strong> Skip connections solve vanishing gradient<br>
            <strong>Depth:</strong> 50-152 layers possible<br>
            <strong>Key Concept:</strong> Residual learning: F(x) + x</p>
            <pre><code>class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual  # Skip connection
        out = self.relu(out)
        return out</code></pre>
            
            <h3>Inception/GoogLeNet (2014)</h3>
            <p><strong>Innovation:</strong> Multi-scale feature extraction<br>
            <strong>Key Idea:</strong> Inception modules with parallel convolutions<br>
            <strong>Parameters:</strong> ~7M (fewer than VGG despite deeper)</p>
            
            <h3>EfficientNet (2019)</h3>
            <p><strong>Innovation:</strong> Compound scaling (depth + width + resolution)<br>
            <strong>Efficiency:</strong> Best accuracy-to-parameters ratio<br>
            <strong>Method:</strong> Neural architecture search + smart scaling</p>
            
            <h3>Performance Comparison (ImageNet Top-5)</h3>
            <p>• AlexNet: 84.6%<br>
            • VGG-16: 92.7%<br>
            • ResNet-50: 92.9%<br>
            • Inception-v3: 93.9%<br>
            • EfficientNet-B7: 96.7%</p>
            
            <h3>Choosing an Architecture</h3>
            <p><strong>For learning:</strong> Start with VGG (simple structure)<br>
            <strong>For performance:</strong> ResNet or EfficientNet<br>
            <strong>For speed:</strong> MobileNet or SqueezeNet<br>
            <strong>For custom tasks:</strong> Transfer learning with pretrained models</p>
            
            <h3>Conclusion</h3>
            <p>Modern CNNs build on these foundational architectures. Understanding their innovations helps in designing custom networks for specific tasks.</p>
        `,
        'note-4': `
            <h2>Understanding LLMs: Large Language Models</h2>
            <h3>What are Large Language Models?</h3>
            <p>Large Language Models (LLMs) are neural networks trained on vast amounts of text data to understand and generate human-like text. They form the backbone of modern AI applications like ChatGPT, Claude, and Gemini.</p>
            
            <h3>Architecture: The Transformer</h3>
            <p>LLMs are built on the transformer architecture introduced in the paper "Attention is All You Need" (2017).</p>
            
            <h3>Key Components</h3>
            <p><strong>1. Self-Attention Mechanism:</strong></p>
            <pre><code>import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        self.queries = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.values = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)
        
    def forward(self, x):
        N, seq_length, _ = x.shape
        
        # Linear projections
        Q = self.queries(x)
        K = self.keys(x)
        V = self.values(x)
        
        # Calculate attention scores
        energy = torch.matmul(Q, K.transpose(-2, -1))
        attention = torch.softmax(energy / (self.head_dim ** 0.5), dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attention, V)
        return self.fc_out(out)</code></pre>
            
            <p><strong>2. Positional Encoding:</strong> Since transformers have no inherent notion of sequence order, positional encodings are added to give the model information about token positions.</p>
            
            <p><strong>3. Feed-Forward Networks:</strong> Each layer has a position-wise feed-forward network applied to each position separately.</p>
            
            <h3>Training LLMs</h3>
            <p><strong>Pre-training:</strong> Unsupervised learning on massive text corpora<br>
            <strong>Objective:</strong> Next token prediction (causal language modeling)<br>
            <strong>Data Scale:</strong> Trillions of tokens<br>
            <strong>Compute:</strong> Thousands of GPUs for months</p>
            
            <h3>Fine-Tuning Techniques</h3>
            <p><strong>Full Fine-Tuning:</strong> Update all parameters (expensive)<br>
            <strong>LoRA (Low-Rank Adaptation):</strong> Add trainable rank decomposition matrices<br>
            <strong>Prompt Tuning:</strong> Learn soft prompts while keeping model frozen<br>
            <strong>RLHF:</strong> Reinforcement Learning from Human Feedback</p>
            
            <h3>LoRA Implementation Example</h3>
            <pre><code>from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

# Load base model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")

# Configure LoRA
lora_config = LoraConfig(
    r=16,  # Rank
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Output: trainable params: 4.7M || all params: 6.7B || trainable%: 0.07%</code></pre>
            
            <h3>Popular LLM Families</h3>
            <p><strong>GPT Series:</strong> OpenAI's autoregressive models<br>
            <strong>Llama:</strong> Meta's open-source LLMs<br>
            <strong>Claude:</strong> Anthropic's constitutional AI<br>
            <strong>Gemini:</strong> Google's multimodal models<br>
            <strong>Mistral:</strong> Efficient open-source models</p>
            
            <h3>Applications</h3>
            <p>• Text generation and completion<br>
            • Question answering and chatbots<br>
            • Code generation and debugging<br>
            • Translation and summarization<br>
            • Creative writing and content creation</p>
            
            <h3>Challenges and Considerations</h3>
            <p><strong>Hallucinations:</strong> Models can generate plausible but incorrect information<br>
            <strong>Bias:</strong> Training data biases reflected in outputs<br>
            <strong>Context Length:</strong> Limited by memory and compute<br>
            <strong>Cost:</strong> Inference and training are expensive</p>
            
            <h3>Future Directions</h3>
            <p>• Longer context windows (100K+ tokens)<br>
            • Multimodal models (text + vision + audio)<br>
            • More efficient architectures<br>
            • Better reasoning and planning capabilities<br>
            • Reduced hallucinations through retrieval augmentation</p>
            
            <h3>Conclusion</h3>
            <p>LLMs represent a paradigm shift in NLP. Understanding their architecture, training, and fine-tuning techniques is essential for modern data science practitioners. The field evolves rapidly—stay updated with latest research!</p>
        `
    };
    
    // Open modal
    viewNoteBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const noteId = btn.getAttribute('data-note-id');
            const noteContent = noteContents[noteId] || `
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
    const projectsGrid = document.querySelector('.completed-projects-grid');
    
    if (!loadMoreBtn || !projectsGrid) return;
    
    // Additional projects data (sorted by date - latest first)
    const additionalProjects = [
        {
            date: '2025-12-01',
            icon: 'fa-database',
            title: 'Customer Churn Prediction Model',
            description: 'Built ML model to predict customer churn with 92% accuracy using ensemble methods and feature engineering.',
            tech: ['Python', 'Scikit-learn', 'XGBoost', 'Pandas'],
            links: [
                { icon: 'fab fa-github', text: 'GitHub', url: 'https://github.com/Labib1610' },
                { icon: 'fas fa-file-pdf', text: 'Report', url: '#' }
            ]
        },
        {
            date: '2025-11-15',
            icon: 'fa-image',
            title: 'Image Classification with ResNet',
            description: 'Implemented transfer learning using ResNet-50 for multi-class image classification achieving 95% accuracy.',
            tech: ['PyTorch', 'ResNet', 'OpenCV', 'Matplotlib'],
            links: [
                { icon: 'fab fa-github', text: 'GitHub', url: 'https://github.com/Labib1610' },
                { icon: 'fas fa-play-circle', text: 'Demo', url: '#' }
            ]
        },
        {
            date: '2025-11-01',
            icon: 'fa-chart-pie',
            title: 'Sales Forecasting Dashboard',
            description: 'Created interactive dashboard with time series forecasting using ARIMA and Prophet models.',
            tech: ['Python', 'Streamlit', 'Prophet', 'Plotly'],
            links: [
                { icon: 'fab fa-github', text: 'GitHub', url: 'https://github.com/Labib1610' },
                { icon: 'fas fa-external-link-alt', text: 'Live', url: '#' }
            ]
        }
    ];
    
    let projectsLoaded = false;
    
    loadMoreBtn.addEventListener('click', () => {
        if (projectsLoaded) {
            // Scroll to top of completed projects if all loaded
            document.getElementById('completed-projects').scrollIntoView({ behavior: 'smooth' });
            return;
        }
        
        // Add new projects
        additionalProjects.forEach(project => {
            const projectCard = document.createElement('div');
            projectCard.className = 'completed-project-card';
            projectCard.style.opacity = '0';
            projectCard.style.transform = 'translateY(20px)';
            
            const linksHTML = project.links.map(link => 
                `<a href="${link.url}" class="project-link" target="_blank">
                    <i class="${link.icon}"></i> ${link.text}
                </a>`
            ).join('\n');
            
            const techHTML = project.tech.map(tech => 
                `<span class="tech-tag">${tech}</span>`
            ).join('\n');
            
            projectCard.innerHTML = `
                <div class="project-header">
                    <div class="project-icon-wrapper">
                        <i class="fas ${project.icon}"></i>
                    </div>
                    <span class="completion-badge"><i class="fas fa-check-circle"></i> Completed</span>
                </div>
                <h3>${project.title}</h3>
                <p>${project.description}</p>
                <div class="project-tech">
                    ${techHTML}
                </div>
                <div class="project-links">
                    ${linksHTML}
                </div>
            `;
            
            projectsGrid.appendChild(projectCard);
            
            // Animate in
            setTimeout(() => {
                projectCard.style.transition = 'all 0.5s ease';
                projectCard.style.opacity = '1';
                projectCard.style.transform = 'translateY(0)';
            }, 100);
        });
        
        projectsLoaded = true;
        loadMoreBtn.innerHTML = '<i class="fas fa-arrow-up"></i> Back to Top';
    });
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
    // Existing functions that are defined
    type();
    setupSmoothScrolling();
    setupMobileNav();
    setupNavbarScroll();
    
    // New functions
    initBackgroundSlideshow();
    initFeaturedCarousel();
    initNotesFilter();
    initNoteModal();
    initLoadMore();
    initAddNote();
    
    console.log('✅ All portfolio features initialized successfully!');
});

// Handle page visibility for animations
document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        // Pause animations when tab is not visible
    } else {
        // Resume animations
    }
});
console.log('Portfolio JS Loaded - Version: 1766778724');
