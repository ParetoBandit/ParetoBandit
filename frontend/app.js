/**
 * BanditGPT - Landing Page Interactive Features
 * Includes: Particle animation, Pareto visualization, Query typing, Demo interactions
 */

// ============================================
// Color Palette (from blog plots)
// ============================================
const COLORS = {
    bg: '#0a0e17',
    panel: '#131a2a',
    grid: '#1e2738',
    text: '#e8eaed',
    muted: '#7a8599',
    accent: '#22c55e',
    gold: '#ffd93d',
    baseline: '#ff6b6b',
    partial: '#f59e0b',
    other: '#2d3748',
    coding: '#22c55e',
    dataScience: '#3b82f6',
    creative: '#a855f7',
    general: '#f59e0b',
};

// ============================================
// Real Model Data (from models_cache.json)
// ============================================
const BASELINE = {
    name: 'Gemini 3 Pro Preview (high)',
    intelligence: 72.8,
    coding: 43.7,
    cost: 2.00,  // $/1M input tokens
    ttft: 32.254,  // seconds
};

const SAMPLE_MODELS = [
    { name: 'Gemini 3 Pro Preview', quality: 100, cost: 2.00, latency: 32254, category: 'baseline', intelligence: 72.8, coding: 43.7 },
    { name: 'GPT-5 (high)', quality: 88, cost: 2.00, latency: 1500, category: 'elite', intelligence: 78.3, coding: 51.0 },
    { name: 'GPT-5.1 (high)', quality: 92, cost: 2.50, latency: 1800, category: 'elite', intelligence: 78.8, coding: 52.1 },
    { name: 'GPT-5 mini (high)', quality: 88, cost: 0.25, latency: 800, category: 'value', intelligence: 64.3, coding: 51.4 },
    { name: 'o4-mini (high)', quality: 85, cost: 0.55, latency: 1200, category: 'value', intelligence: 68.5, coding: 53.2 },
    { name: 'Grok 4.1 Fast (Reasoning)', quality: 88, cost: 0.20, latency: 15936, category: 'value', intelligence: 64.1, coding: 49.7 },
    { name: 'Grok 4 Fast (Reasoning)', quality: 84, cost: 0.20, latency: 3387, category: 'value', intelligence: 60.3, coding: 48.4 },
    { name: 'gpt-oss-120B (high)', quality: 81, cost: 0.26, latency: 2500, category: 'value', intelligence: 58.9, coding: 48.0 },
    { name: 'DeepSeek V3.1 (Reasoning)', quality: 86, cost: 0.14, latency: 2100, category: 'budget', intelligence: 63.2, coding: 42.0 },
    { name: 'DeepSeek V3.2 Exp', quality: 88, cost: 0.14, latency: 1900, category: 'budget', intelligence: 64.0, coding: 43.5 },
    { name: 'Qwen3 32B (Reasoning)', quality: 75, cost: 0.10, latency: 1500, category: 'budget', intelligence: 54.3, coding: 38.2 },
    { name: 'Claude 4.5 Sonnet (Reasoning)', quality: 95, cost: 9.00, latency: 2800, category: 'premium', intelligence: 76.2, coding: 49.8 },
    { name: 'Gemini 2.5 Pro', quality: 94, cost: 2.50, latency: 3500, category: 'general', intelligence: 68.1, coding: 45.2 },
];

// Query examples with real HYBRID optimization results
const QUERY_EXAMPLES = [
    {
        text: "For CODING tasks, give me 85% quality of the leading model at low cost",
        task: { label: "💻 Coding", class: "coding" },
        result: { name: "GPT-5 mini (high)", stats: "Q:88% | $0.25/M | 0.3s TTFT" },
        baseline: BASELINE,
        recommended: { name: "GPT-5 mini (high)", cost: 0.25, ttft: 0.3, quality: 88 }
    },
    {
        text: "Best value model for DATA SCIENCE with good math capabilities",
        task: { label: "📊 Data Science", class: "data-science" },
        result: { name: "GPT-5 mini (high)", stats: "Q:88% | $0.25/M | 0.3s TTFT" },
        baseline: BASELINE,
        recommended: { name: "GPT-5 mini (high)", cost: 0.25, ttft: 0.3, quality: 88 }
    },
    {
        text: "Optimize for CREATIVE writing with maximum cost savings",
        task: { label: "✨ Creative", class: "creative" },
        result: { name: "Grok 4.1 Fast (Reasoning)", stats: "Q:86% | $0.20/M | 15.9s TTFT" },
        baseline: BASELINE,
        recommended: { name: "Grok 4.1 Fast (Reasoning)", cost: 0.20, ttft: 15.9, quality: 86 }
    },
    {
        text: "Best GENERAL assistant with high intelligence score",
        task: { label: "🎯 General", class: "general" },
        result: { name: "GPT-5 (high)", stats: "Q:92% | $1.25/M | 0.3s TTFT" },
        baseline: BASELINE,
        recommended: { name: "GPT-5 (high)", cost: 1.25, ttft: 0.3, quality: 92 }
    },
];

// ============================================
// Particle Animation
// ============================================
class ParticleAnimation {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        if (!this.canvas) return;
        
        this.ctx = this.canvas.getContext('2d');
        this.particles = [];
        this.mouse = { x: null, y: null };
        
        this.resize();
        this.init();
        this.animate();
        
        window.addEventListener('resize', () => this.resize());
        window.addEventListener('mousemove', (e) => {
            this.mouse.x = e.clientX;
            this.mouse.y = e.clientY;
        });
    }
    
    resize() {
        this.canvas.width = window.innerWidth;
        this.canvas.height = window.innerHeight;
    }
    
    init() {
        this.particles = [];
        const numParticles = Math.floor((this.canvas.width * this.canvas.height) / 15000);
        
        for (let i = 0; i < numParticles; i++) {
            this.particles.push({
                x: Math.random() * this.canvas.width,
                y: Math.random() * this.canvas.height,
                vx: (Math.random() - 0.5) * 0.3,
                vy: (Math.random() - 0.5) * 0.3,
                radius: Math.random() * 2 + 1,
                color: this.getRandomColor(),
            });
        }
    }
    
    getRandomColor() {
        const colors = [COLORS.accent, COLORS.dataScience, COLORS.creative, COLORS.gold];
        return colors[Math.floor(Math.random() * colors.length)];
    }
    
    animate() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        this.particles.forEach(p => {
            // Update position
            p.x += p.vx;
            p.y += p.vy;
            
            // Wrap around edges
            if (p.x < 0) p.x = this.canvas.width;
            if (p.x > this.canvas.width) p.x = 0;
            if (p.y < 0) p.y = this.canvas.height;
            if (p.y > this.canvas.height) p.y = 0;
            
            // Draw particle
            this.ctx.beginPath();
            this.ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
            this.ctx.fillStyle = p.color + '40';
            this.ctx.fill();
        });
        
        // Draw connections
        this.particles.forEach((p1, i) => {
            this.particles.slice(i + 1).forEach(p2 => {
                const dx = p1.x - p2.x;
                const dy = p1.y - p2.y;
                const distance = Math.sqrt(dx * dx + dy * dy);
                
                if (distance < 120) {
                    this.ctx.beginPath();
                    this.ctx.moveTo(p1.x, p1.y);
                    this.ctx.lineTo(p2.x, p2.y);
                    this.ctx.strokeStyle = COLORS.accent + Math.floor((1 - distance / 120) * 30).toString(16).padStart(2, '0');
                    this.ctx.lineWidth = 0.5;
                    this.ctx.stroke();
                }
            });
        });
        
        requestAnimationFrame(() => this.animate());
    }
}

// ============================================
// Typing Animation for Query Demo
// ============================================
class QueryTypingAnimation {
    constructor() {
        this.element = document.getElementById('animated-query');
        this.resultElement = document.getElementById('query-result');
        this.modelNameElement = document.getElementById('result-model-name');
        this.statsElement = document.getElementById('result-stats');
        this.taskTagElement = document.getElementById('result-task-tag');
        this.savingsElement = document.getElementById('hero-savings');
        
        // Savings display elements
        this.savingsCost = document.getElementById('savings-cost');
        this.savingsCostDetail = document.getElementById('savings-cost-detail');
        this.savingsLatency = document.getElementById('savings-latency');
        this.savingsLatencyDetail = document.getElementById('savings-latency-detail');
        this.savingsValue = document.getElementById('savings-value');
        
        if (!this.element) return;
        
        this.currentIndex = 0;
        this.charIndex = 0;
        this.isDeleting = false;
        this.isPaused = false;
        
        this.animate();
    }
    
    animate() {
        const current = QUERY_EXAMPLES[this.currentIndex];
        const fullText = current.text;
        
        if (!this.isDeleting && !this.isPaused) {
            // Typing
            this.element.textContent = fullText.substring(0, this.charIndex + 1);
            this.charIndex++;
            
            if (this.charIndex === fullText.length) {
                // Finished typing, show result, task tag, and savings all together
                this.showResult(current.result, current.task);
                this.updateSavings(current.baseline, current.recommended);
                this.isPaused = true;
                // Longer pause (5 seconds) for users to read the result
                setTimeout(() => {
                    this.isPaused = false;
                    this.isDeleting = true;
                    this.animate();
                }, 5000);
                return;
            }
            
            setTimeout(() => this.animate(), 60 + Math.random() * 40);
            
        } else if (this.isDeleting) {
            // Deleting
            this.element.textContent = fullText.substring(0, this.charIndex);
            this.charIndex--;
            
            if (this.charIndex === 0) {
                this.isDeleting = false;
                this.hideResult();
                this.currentIndex = (this.currentIndex + 1) % QUERY_EXAMPLES.length;
                setTimeout(() => this.animate(), 500);
                return;
            }
            
            setTimeout(() => this.animate(), 25);
        }
    }
    
    showResult(result, task) {
        // Update task tag
        if (this.taskTagElement && task) {
            this.taskTagElement.textContent = task.label;
            this.taskTagElement.className = 'result-task-tag ' + task.class;
        }
        
        // Update model info
        this.modelNameElement.textContent = result.name;
        this.statsElement.textContent = result.stats;
        this.resultElement.classList.add('visible');
        
        // Show savings immediately (no delay)
        if (this.savingsElement) {
            this.savingsElement.classList.add('visible');
        }
    }
    
    hideResult() {
        this.resultElement.classList.remove('visible');
        if (this.savingsElement) {
            this.savingsElement.classList.remove('visible');
        }
    }
    
    updateSavings(baseline, recommended) {
        if (!this.savingsElement) return;
        
        // Calculate cost savings
        const costSavings = ((baseline.cost - recommended.cost) / baseline.cost) * 100;
        const costSaved = baseline.cost - recommended.cost;
        
        // Calculate latency improvement
        const latencyImprovement = ((baseline.ttft - recommended.ttft) / baseline.ttft) * 100;
        
        // Calculate value score (quality per dollar)
        const baselineValue = 100 / baseline.cost;  // baseline quality is 100%
        const recommendedValue = recommended.quality / recommended.cost;
        const valueMultiplier = recommendedValue / baselineValue;
        
        // Update the display
        if (this.savingsCost) {
            this.savingsCost.textContent = costSavings.toFixed(1) + '%';
        }
        if (this.savingsCostDetail) {
            this.savingsCostDetail.textContent = `($${costSaved.toFixed(2)} saved per 1M tokens)`;
        }
        if (this.savingsLatency) {
            const sign = latencyImprovement > 0 ? '-' : '+';
            this.savingsLatency.textContent = sign + Math.abs(latencyImprovement).toFixed(0) + '%';
        }
        if (this.savingsLatencyDetail) {
            this.savingsLatencyDetail.textContent = `(${recommended.ttft.toFixed(1)}s vs ${baseline.ttft.toFixed(1)}s TTFT)`;
        }
        if (this.savingsValue) {
            this.savingsValue.textContent = valueMultiplier.toFixed(1) + 'x';
        }
        // Note: visibility is now handled in showResult() for proper sync
    }
}

// ============================================
// Top 10 HYBRID Rankings per Use Case (from real optimizer)
// ============================================
const HYBRID_TOP_10 = {
    "coding": [
        {
            "name": "GPT-5 mini (high)",
            "quality": 86.1,
            "cost": 0.69,
            "ttft": 0.34,
            "cost_pct": 20.0,
            "ttft_pct": 102.9
        },
        {
            "name": "Grok 4.1 Fast (Reasoning)",
            "quality": 84.2,
            "cost": 0.28,
            "ttft": 0.63,
            "cost_pct": 8.0,
            "ttft_pct": 190.4
        },
        {
            "name": "gpt-oss-120B (high)",
            "quality": 83.6,
            "cost": 0.26,
            "ttft": 0.32,
            "cost_pct": 7.6,
            "ttft_pct": 94.9
        },
        {
            "name": "DeepSeek V3.1 Terminus (Reasoning)",
            "quality": 77.5,
            "cost": 0.8,
            "ttft": 0.44,
            "cost_pct": 23.3,
            "ttft_pct": 132.7
        },
        {
            "name": "o4-mini (high)",
            "quality": 85.8,
            "cost": 1.93,
            "ttft": 0.26,
            "cost_pct": 56.0,
            "ttft_pct": 79.7
        },
        {
            "name": "DeepSeek V3.2 Exp (Reasoning)",
            "quality": 71.5,
            "cost": 0.32,
            "ttft": 0.57,
            "cost_pct": 9.2,
            "ttft_pct": 172.1
        },
        {
            "name": "GPT-5.1 (high)",
            "quality": 100.0,
            "cost": 3.44,
            "ttft": 0.33,
            "cost_pct": 100.0,
            "ttft_pct": 100.0
        },
        {
            "name": "Grok 3 mini Reasoning (high)",
            "quality": 67.3,
            "cost": 0.35,
            "ttft": 0.51,
            "cost_pct": 10.2,
            "ttft_pct": 152.8
        },
        {
            "name": "Gemini 2.5 Flash Preview (Sep '25) (Reasoning)",
            "quality": 66.4,
            "cost": 0.85,
            "ttft": 0.63,
            "cost_pct": 24.7,
            "ttft_pct": 188.8
        },
        {
            "name": "GLM-4.6 (Reasoning)",
            "quality": 66.1,
            "cost": 1,
            "ttft": 1.69,
            "cost_pct": 29.1,
            "ttft_pct": 508.3
        }
    ],
    "data_science": [
        {
            "name": "GPT-5 mini (high)",
            "quality": 85.8,
            "cost": 0.69,
            "ttft": 0.34,
            "cost_pct": 20.0,
            "ttft_pct": 102.9
        },
        {
            "name": "Grok 4.1 Fast (Reasoning)",
            "quality": 83.6,
            "cost": 0.28,
            "ttft": 0.63,
            "cost_pct": 8.0,
            "ttft_pct": 190.4
        },
        {
            "name": "gpt-oss-120B (high)",
            "quality": 81.7,
            "cost": 0.26,
            "ttft": 0.32,
            "cost_pct": 7.6,
            "ttft_pct": 94.9
        },
        {
            "name": "DeepSeek V3.1 Terminus (Reasoning)",
            "quality": 77.5,
            "cost": 0.8,
            "ttft": 0.44,
            "cost_pct": 23.3,
            "ttft_pct": 132.7
        },
        {
            "name": "DeepSeek V3.2 Exp (Reasoning)",
            "quality": 71.6,
            "cost": 0.32,
            "ttft": 0.57,
            "cost_pct": 9.2,
            "ttft_pct": 172.1
        },
        {
            "name": "Grok 3 mini Reasoning (high)",
            "quality": 71.3,
            "cost": 0.35,
            "ttft": 0.51,
            "cost_pct": 10.2,
            "ttft_pct": 152.8
        },
        {
            "name": "o4-mini (high)",
            "quality": 82.2,
            "cost": 1.93,
            "ttft": 0.26,
            "cost_pct": 56.0,
            "ttft_pct": 79.7
        },
        {
            "name": "GPT-5.1 (high)",
            "quality": 100.0,
            "cost": 3.44,
            "ttft": 0.33,
            "cost_pct": 100.0,
            "ttft_pct": 100.0
        },
        {
            "name": "GLM-4.6 (Reasoning)",
            "quality": 67.5,
            "cost": 1,
            "ttft": 1.69,
            "cost_pct": 29.1,
            "ttft_pct": 508.3
        },
        {
            "name": "Gemini 2.5 Flash Preview (Sep '25) (Reasoning)",
            "quality": 65.1,
            "cost": 0.85,
            "ttft": 0.63,
            "cost_pct": 24.7,
            "ttft_pct": 188.8
        }
    ],
    "creative": [
        {
            "name": "Grok 4.1 Fast (Reasoning)",
            "quality": 82.6,
            "cost": 0.28,
            "ttft": 0.63,
            "cost_pct": 2.8,
            "ttft_pct": 36.9
        },
        {
            "name": "GPT-5 mini (high)",
            "quality": 80.8,
            "cost": 0.69,
            "ttft": 0.34,
            "cost_pct": 6.9,
            "ttft_pct": 19.9
        },
        {
            "name": "DeepSeek V3.1 Terminus (Reasoning)",
            "quality": 77.6,
            "cost": 0.8,
            "ttft": 0.44,
            "cost_pct": 8.0,
            "ttft_pct": 25.7
        },
        {
            "name": "GPT-5.1 (high)",
            "quality": 97.2,
            "cost": 3.44,
            "ttft": 0.33,
            "cost_pct": 34.4,
            "ttft_pct": 19.4
        },
        {
            "name": "DeepSeek V3.2 Exp (Reasoning)",
            "quality": 76.2,
            "cost": 0.32,
            "ttft": 0.57,
            "cost_pct": 3.1,
            "ttft_pct": 33.3
        },
        {
            "name": "Grok 3 mini Reasoning (high)",
            "quality": 70.9,
            "cost": 0.35,
            "ttft": 0.51,
            "cost_pct": 3.5,
            "ttft_pct": 29.6
        },
        {
            "name": "Gemini 2.5 Flash Preview (Sep '25) (Reasoning)",
            "quality": 70.1,
            "cost": 0.85,
            "ttft": 0.63,
            "cost_pct": 8.5,
            "ttft_pct": 36.5
        },
        {
            "name": "Gemini 3 Pro Preview (high)",
            "quality": 102.7,
            "cost": 4.5,
            "ttft": 1.96,
            "cost_pct": 45.0,
            "ttft_pct": 114.5
        },
        {
            "name": "gpt-oss-120B (high)",
            "quality": 68.5,
            "cost": 0.26,
            "ttft": 0.32,
            "cost_pct": 2.6,
            "ttft_pct": 18.4
        },
        {
            "name": "GLM-4.6 (Reasoning)",
            "quality": 68.2,
            "cost": 1,
            "ttft": 1.69,
            "cost_pct": 10.0,
            "ttft_pct": 98.4
        }
    ],
    "general": [
        {
            "name": "DeepSeek V3.1 Terminus (Reasoning)",
            "quality": 87.0,
            "cost": 0.8,
            "ttft": 0.44,
            "cost_pct": 23.3,
            "ttft_pct": 132.7
        },
        {
            "name": "GPT-5 mini (high)",
            "quality": 85.2,
            "cost": 0.69,
            "ttft": 0.34,
            "cost_pct": 20.0,
            "ttft_pct": 102.9
        },
        {
            "name": "DeepSeek V3.2 Exp (Reasoning)",
            "quality": 83.8,
            "cost": 0.32,
            "ttft": 0.57,
            "cost_pct": 9.2,
            "ttft_pct": 172.1
        },
        {
            "name": "Grok 4.1 Fast (Reasoning)",
            "quality": 81.0,
            "cost": 0.28,
            "ttft": 0.63,
            "cost_pct": 8.0,
            "ttft_pct": 190.4
        },
        {
            "name": "GPT-5.1 (high)",
            "quality": 100.0,
            "cost": 3.44,
            "ttft": 0.33,
            "cost_pct": 100.0,
            "ttft_pct": 100.0
        },
        {
            "name": "Gemini 2.5 Pro",
            "quality": 88.1,
            "cost": 3.44,
            "ttft": 0.94,
            "cost_pct": 100.0,
            "ttft_pct": 284.2
        },
        {
            "name": "Grok 3 mini Reasoning (high)",
            "quality": 80.2,
            "cost": 0.35,
            "ttft": 0.51,
            "cost_pct": 10.2,
            "ttft_pct": 152.8
        },
        {
            "name": "gpt-oss-120B (high)",
            "quality": 76.6,
            "cost": 0.26,
            "ttft": 0.32,
            "cost_pct": 7.6,
            "ttft_pct": 94.9
        },
        {
            "name": "o4-mini (high)",
            "quality": 75.7,
            "cost": 1.93,
            "ttft": 0.26,
            "cost_pct": 56.0,
            "ttft_pct": 79.7
        },
        {
            "name": "Gemini 3 Pro Preview (high)",
            "quality": 103.3,
            "cost": 4.5,
            "ttft": 1.96,
            "cost_pct": 130.9,
            "ttft_pct": 591.5
        }
    ],
    "qa": [
        {
            "name": "GPT-5.1 (high)",
            "quality": 98.7,
            "cost": 3.44,
            "ttft": 0.33,
            "cost_pct": 34.4,
            "ttft_pct": 19.4
        },
        {
            "name": "DeepSeek V3.1 Terminus (Reasoning)",
            "quality": 93.1,
            "cost": 0.8,
            "ttft": 0.44,
            "cost_pct": 8.0,
            "ttft_pct": 25.7
        },
        {
            "name": "DeepSeek V3.2 Exp (Reasoning)",
            "quality": 91.4,
            "cost": 0.32,
            "ttft": 0.57,
            "cost_pct": 3.1,
            "ttft_pct": 33.3
        },
        {
            "name": "Gemini 2.5 Pro",
            "quality": 93.6,
            "cost": 3.44,
            "ttft": 0.94,
            "cost_pct": 34.4,
            "ttft_pct": 55.0
        },
        {
            "name": "Grok 3 mini Reasoning (high)",
            "quality": 88.0,
            "cost": 0.35,
            "ttft": 0.51,
            "cost_pct": 3.5,
            "ttft_pct": 29.6
        },
        {
            "name": "Gemini 2.5 Flash-Lite Preview (Sep '25) (Reasoning)",
            "quality": 85.1,
            "cost": 0.17,
            "ttft": 0.51,
            "cost_pct": 1.7,
            "ttft_pct": 29.8
        },
        {
            "name": "Gemini 3 Pro Preview (high)",
            "quality": 99.5,
            "cost": 4.5,
            "ttft": 1.96,
            "cost_pct": 45.0,
            "ttft_pct": 114.5
        },
        {
            "name": "GPT-5 mini (high)",
            "quality": 84.3,
            "cost": 0.69,
            "ttft": 0.34,
            "cost_pct": 6.9,
            "ttft_pct": 19.9
        },
        {
            "name": "Gemini 2.5 Flash Preview (Sep '25) (Reasoning)",
            "quality": 79.7,
            "cost": 0.85,
            "ttft": 0.63,
            "cost_pct": 8.5,
            "ttft_pct": 36.5
        },
        {
            "name": "Grok 4.1 Fast (Reasoning)",
            "quality": 79.2,
            "cost": 0.28,
            "ttft": 0.63,
            "cost_pct": 2.8,
            "ttft_pct": 36.9
        }
    ],
    "rag": [
        {
            "name": "Grok 4.1 Fast (Reasoning)",
            "quality": 87.6,
            "cost": 0.28,
            "ttft": 0.63,
            "cost_pct": 6.1,
            "ttft_pct": 32.2,
            "context_k": 2000
        },
        {
            "name": "Gemini 2.5 Flash-Lite Preview (Sep '25) (Reasoning)",
            "quality": 82.8,
            "cost": 0.17,
            "ttft": 0.51,
            "cost_pct": 3.9,
            "ttft_pct": 26.0,
            "context_k": 1048
        },
        {
            "name": "GPT-5 mini (high)",
            "quality": 81.0,
            "cost": 0.69,
            "ttft": 0.34,
            "cost_pct": 15.3,
            "ttft_pct": 17.4,
            "context_k": 400
        },
        {
            "name": "Gemini 2.5 Pro",
            "quality": 93.2,
            "cost": 3.44,
            "ttft": 0.94,
            "cost_pct": 76.4,
            "ttft_pct": 48.0,
            "context_k": 1048
        },
        {
            "name": "GPT-5.1 (high)",
            "quality": 90.5,
            "cost": 3.44,
            "ttft": 0.33,
            "cost_pct": 76.4,
            "ttft_pct": 16.9,
            "context_k": 400
        },
        {
            "name": "DeepSeek V3.2 Exp (Reasoning)",
            "quality": 74.1,
            "cost": 0.32,
            "ttft": 0.57,
            "cost_pct": 7.0,
            "ttft_pct": 29.1,
            "context_k": 163
        },
        {
            "name": "DeepSeek V3.1 Terminus (Reasoning)",
            "quality": 72.3,
            "cost": 0.8,
            "ttft": 0.44,
            "cost_pct": 17.8,
            "ttft_pct": 22.4,
            "context_k": 131
        },
        {
            "name": "GLM-4.6 (Reasoning)",
            "quality": 70.0,
            "cost": 1,
            "ttft": 1.69,
            "cost_pct": 22.2,
            "ttft_pct": 85.9,
            "context_k": 202
        },
        {
            "name": "Gemini 3 Pro Preview (high)",
            "quality": 100.0,
            "cost": 4.5,
            "ttft": 1.96,
            "cost_pct": 100.0,
            "ttft_pct": 100.0,
            "context_k": 1048
        },
        {
            "name": "Grok 3 mini Reasoning (high)",
            "quality": 68.6,
            "cost": 0.35,
            "ttft": 0.51,
            "cost_pct": 7.8,
            "ttft_pct": 25.8,
            "context_k": 131
        }
    ],
    "chatbot": [
        {
            "name": "Grok 4.1 Fast (Reasoning)",
            "quality": 96.6,
            "cost": 0.28,
            "ttft": 0.63,
            "cost_pct": 40.0,
            "ttft_pct": 185.0
        },
        {
            "name": "DeepSeek V3.2 Exp (Reasoning)",
            "quality": 96.3,
            "cost": 0.32,
            "ttft": 0.57,
            "cost_pct": 45.8,
            "ttft_pct": 167.2
        },
        {
            "name": "GPT-5 mini (high)",
            "quality": 100.0,
            "cost": 0.69,
            "ttft": 0.34,
            "cost_pct": 100.0,
            "ttft_pct": 100.0
        },
        {
            "name": "Grok 3 mini Reasoning (high)",
            "quality": 90.5,
            "cost": 0.35,
            "ttft": 0.51,
            "cost_pct": 50.9,
            "ttft_pct": 148.5
        },
        {
            "name": "DeepSeek V3.1 Terminus (Reasoning)",
            "quality": 99.1,
            "cost": 0.8,
            "ttft": 0.44,
            "cost_pct": 116.3,
            "ttft_pct": 128.9
        },
        {
            "name": "gpt-oss-120B (high)",
            "quality": 88.0,
            "cost": 0.26,
            "ttft": 0.32,
            "cost_pct": 38.2,
            "ttft_pct": 92.2
        },
        {
            "name": "Gemini 2.5 Flash Preview (Sep '25) (Reasoning)",
            "quality": 86.5,
            "cost": 0.85,
            "ttft": 0.63,
            "cost_pct": 123.5,
            "ttft_pct": 183.4
        },
        {
            "name": "GLM-4.6 (Reasoning)",
            "quality": 85.4,
            "cost": 1,
            "ttft": 1.69,
            "cost_pct": 145.3,
            "ttft_pct": 493.9
        },
        {
            "name": "Gemini 2.5 Flash-Lite Preview (Sep '25) (Reasoning)",
            "quality": 80.2,
            "cost": 0.17,
            "ttft": 0.51,
            "cost_pct": 25.4,
            "ttft_pct": 149.6
        },
        {
            "name": "GLM-4.5-Air",
            "quality": 75.0,
            "cost": 0.42,
            "ttft": 0.69,
            "cost_pct": 61.8,
            "ttft_pct": 201.9
        }
    ]
};

// Slider value mappings
const QUALITY_VALUES = [70, 80, 90];
const COST_VALUES = [25, 50, 100];
const LATENCY_VALUES = [10, 50, 100];

// Open source model identification
const OPEN_SOURCE_PATTERNS = [
    'DeepSeek',
    'Qwen',
    'GLM',
    'Llama',
    'Mistral',
    'Gemma',
    'Phi-',
];

function isOpenSource(modelName) {
    return OPEN_SOURCE_PATTERNS.some(pattern => modelName.includes(pattern));
}

// Use-case specific baseline models (the "premium incumbent" for comparison)
const USE_CASE_BASELINES = {
    'coding': 'GPT-5.1 (high)',
    'data_science': 'GPT-5.1 (high)',
    'creative': 'Claude Opus 4.5',
    'general': 'GPT-5.1 (high)',
    'qa': 'Claude Opus 4.5',
    'rag': 'Gemini 3 Pro Preview',
    'chatbot': 'GPT-5 mini (high)',
};

// ============================================
// Interactive Demo (showing top 10 with constraint indicators)
// ============================================
class InteractiveDemo {
    constructor() {
        this.qualitySlider = document.getElementById('quality-slider');
        this.costSlider = document.getElementById('cost-slider');
        this.latencySlider = document.getElementById('latency-slider');
        this.qualityValue = document.getElementById('quality-value');
        this.costValue = document.getElementById('cost-value');
        this.latencyValue = document.getElementById('latency-value');
        this.resultsList = document.getElementById('results-list');
        this.resultsCount = document.getElementById('results-count');
        this.opensourceCheckbox = document.getElementById('opensource-only');
        this.baselineModel = document.getElementById('baseline-model');
        
        if (!this.qualitySlider) return;
        
        this.useCase = 'coding';
        this.opensourceOnly = false;
        
        this.setupEventListeners();
        this.updateBaselineDisplay();
        this.updateResults();
    }
    
    updateBaselineDisplay() {
        if (this.baselineModel) {
            this.baselineModel.textContent = USE_CASE_BASELINES[this.useCase] || 'GPT-5.1 (high)';
        }
    }
    
    setupEventListeners() {
        // Quality slider (discrete: 0=70%, 1=80%, 2=90%)
        this.qualitySlider.addEventListener('input', () => {
            const idx = parseInt(this.qualitySlider.value);
            this.qualityValue.textContent = QUALITY_VALUES[idx] + '%';
            this.updateResults();
        });
        
        // Cost slider (discrete: 0=25%, 1=50%, 2=100%)
        this.costSlider.addEventListener('input', () => {
            const idx = parseInt(this.costSlider.value);
            this.costValue.textContent = COST_VALUES[idx] + '%';
            this.updateResults();
        });
        
        // Latency slider (discrete: 0=10%, 1=50%, 2=100%)
        if (this.latencySlider) {
            this.latencySlider.addEventListener('input', () => {
                const idx = parseInt(this.latencySlider.value);
                this.latencyValue.textContent = LATENCY_VALUES[idx] + '%';
                this.updateResults();
            });
        }
        
        // Open source checkbox
        if (this.opensourceCheckbox) {
            this.opensourceCheckbox.addEventListener('change', () => {
                this.opensourceOnly = this.opensourceCheckbox.checked;
                this.updateResults();
            });
        }
        
        // Use case buttons (only in constraints mode, not budget mode)
        document.querySelectorAll('#constraints-mode .use-case-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('#constraints-mode .use-case-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                this.useCase = btn.dataset.usecase;
                this.updateBaselineDisplay();
                this.updateResults();
            });
        });
    }
    
    updateResults() {
        const qualityIdx = parseInt(this.qualitySlider.value);
        const costIdx = parseInt(this.costSlider.value);
        const latencyIdx = this.latencySlider ? parseInt(this.latencySlider.value) : 2;
        
        const minQuality = QUALITY_VALUES[qualityIdx];
        const maxCostPct = COST_VALUES[costIdx];
        const maxLatencyPct = LATENCY_VALUES[latencyIdx];
        
        // Get top 10 for this use case
        let allModels = HYBRID_TOP_10[this.useCase] || [];
        
        // Filter by open source if checkbox is checked
        if (this.opensourceOnly) {
            allModels = allModels.filter(model => isOpenSource(model.name));
        }
        
        // Check constraints for each model and preserve original rank
        const results = allModels.map((model, originalRank) => {
            const meetsQuality = model.quality >= minQuality;
            const meetsCost = model.cost_pct <= maxCostPct;
            const meetsLatency = model.ttft_pct <= maxLatencyPct;
            const meetsAll = meetsQuality && meetsCost && meetsLatency;
            const constraintScore = (meetsQuality ? 1 : 0) + (meetsCost ? 1 : 0) + (meetsLatency ? 1 : 0);
            return { ...model, meetsQuality, meetsCost, meetsLatency, meetsAll, constraintScore, originalRank };
        });
        
        // Sort by: 1) number of constraints met (desc), 2) original HYBRID rank (asc)
        results.sort((a, b) => {
            if (b.constraintScore !== a.constraintScore) {
                return b.constraintScore - a.constraintScore; // More constraints met = higher
            }
            return a.originalRank - b.originalRank; // Preserve HYBRID order within group
        });
        
        // Count how many meet all constraints
        const matchCount = results.filter(r => r.meetsAll).length;
        this.resultsCount.textContent = `${matchCount} of ${results.length} meet constraints`;
        
        // Render all results with constraint indicators
        this.resultsList.innerHTML = results.map((model, index) => {
            // Determine visual state
            let stateClass = '';
            let stateIcon = '';
            if (model.meetsAll) {
                stateClass = 'meets-all';
                stateIcon = '✓';
            } else if (model.meetsQuality || model.meetsCost || model.meetsLatency) {
                stateClass = 'meets-partial';
                stateIcon = '~';
            } else {
                stateClass = 'meets-none';
                stateIcon = '✗';
            }
            
            // Show which constraint failed
            let constraintTags = '';
            if (!model.meetsAll) {
                if (!model.meetsQuality) {
                    constraintTags += `<span class="constraint-tag fail">Q &lt; ${minQuality}%</span>`;
                }
                if (!model.meetsCost) {
                    constraintTags += `<span class="constraint-tag fail">C &gt; ${maxCostPct}%</span>`;
                }
                if (!model.meetsLatency) {
                    constraintTags += `<span class="constraint-tag fail">L &gt; ${maxLatencyPct}%</span>`;
                }
            }
            
            // Show rank with trophy for #1 if it meets all constraints
            const rankDisplay = index === 0 && model.meetsAll ? '🏆' : (index + 1);
            
            const isOSS = isOpenSource(model.name);
            const ossBadge = isOSS ? '<span class="oss-badge">OSS</span>' : '';
            
            return `
                <div class="result-item ${stateClass} ${index === 0 && model.meetsAll ? 'winner' : ''}">
                    <div class="result-rank ${stateClass}">${rankDisplay}</div>
                    <div class="result-info">
                        <div class="result-name-row">
                            <span class="result-model-name">${model.name}</span>
                            ${ossBadge}
                            <span class="constraint-status ${stateClass}">${stateIcon}</span>
                        </div>
                        <span class="result-metrics">
                            <span class="metric ${model.meetsQuality ? 'pass' : 'fail'}">Q:${model.quality.toFixed(0)}%</span> | 
                            <span class="metric ${model.meetsCost ? 'pass' : 'fail'}">C:${model.cost_pct.toFixed(0)}%</span> | 
                            <span class="metric ${model.meetsLatency ? 'pass' : 'fail'}">L:${model.ttft_pct.toFixed(0)}%</span> | 
                            ${model.ttft.toFixed(2)}s
                            ${model.context_k ? ` | <span class="metric context">📄 ${model.context_k}K ctx</span>` : ''}
                        </span>
                        ${constraintTags ? `<div class="constraint-tags">${constraintTags}</div>` : ''}
                    </div>
                    <div class="result-savings">
                        <span class="savings-badge cost-badge">↓${(100 - model.cost_pct).toFixed(0)}% cost</span>
                        ${model.ttft_pct <= 100 
                            ? `<span class="savings-badge latency-badge">↓${(100 - model.ttft_pct).toFixed(0)}% faster</span>`
                            : `<span class="savings-badge latency-badge slower">↑${(model.ttft_pct - 100).toFixed(0)}% slower</span>`
                        }
                    </div>
                </div>
            `;
        }).join('');
    }
}

// ============================================
// Budget Mode Demo (using real HYBRID rankings)
// ============================================

// Budget values in $/M tokens
const BUDGET_VALUES = [0.10, 0.25, 0.50, 1.00, 2.00, 5.00];

// Pre-computed HYBRID rankings for each budget level and use case
const BUDGET_RANKINGS = {
    "coding_0_1": [
        {
            "name": "Qwen3 4B 2507 (Reasoning)",
            "quality": 54.7,
            "cost": 0.0,
            "ttft": 0.91,
            "cost_pct": 0.0,
            "ttft_pct": 273.0,
            "score": 0.2
        },
        {
            "name": "DeepSeek R1 0528 Qwen3 8B",
            "quality": 44.9,
            "cost": 0.07,
            "ttft": 0.77,
            "cost_pct": 2.0,
            "ttft_pct": 232.3,
            "score": 0.2058
        },
        {
            "name": "Gemma 3 27B Instruct",
            "quality": 21.3,
            "cost": 0.0,
            "ttft": 0.41,
            "cost_pct": 0.0,
            "ttft_pct": 122.3,
            "score": 0.2817
        },
        {
            "name": "Gemma 3 12B Instruct",
            "quality": 17.2,
            "cost": 0.0,
            "ttft": 0.44,
            "cost_pct": 0.0,
            "ttft_pct": 132.7,
            "score": 0.2948
        },
        {
            "name": "Phi-4 Mini Instruct",
            "quality": 8.8,
            "cost": 0.0,
            "ttft": 0.7,
            "cost_pct": 0.0,
            "ttft_pct": 211.8,
            "score": 0.3217
        },
        {
            "name": "Ministral 8B",
            "quality": 6.9,
            "cost": 0.1,
            "ttft": 0.72,
            "cost_pct": 2.9,
            "ttft_pct": 215.6,
            "score": 0.3279
        },
        {
            "name": "Gemma 3 4B Instruct",
            "quality": 6.2,
            "cost": 0.0,
            "ttft": 0.29,
            "cost_pct": 0.0,
            "ttft_pct": 86.7,
            "score": 0.33
        },
        {
            "name": "Ministral 3B",
            "quality": 1.5,
            "cost": 0.04,
            "ttft": 0.53,
            "cost_pct": 1.2,
            "ttft_pct": 158.6,
            "score": 0.3453
        }
    ],
    "coding_0_25": [
        {
            "name": "GPT-5 nano (high)",
            "quality": 64.7,
            "cost": 0.14,
            "ttft": 0.27,
            "cost_pct": 4.0,
            "ttft_pct": 80.5,
            "score": 0.1426
        },
        {
            "name": "Gemini 2.5 Flash-Lite Preview (Sep '25) (Reasoning)",
            "quality": 56.7,
            "cost": 0.17,
            "ttft": 0.51,
            "cost_pct": 5.1,
            "ttft_pct": 154.0,
            "score": 0.1683
        },
        {
            "name": "Qwen3 4B 2507 (Reasoning)",
            "quality": 54.7,
            "cost": 0.0,
            "ttft": 0.91,
            "cost_pct": 0.0,
            "ttft_pct": 273.0,
            "score": 0.1747
        },
        {
            "name": "DeepSeek R1 0528 Qwen3 8B",
            "quality": 44.9,
            "cost": 0.07,
            "ttft": 0.77,
            "cost_pct": 2.0,
            "ttft_pct": 232.3,
            "score": 0.2058
        },
        {
            "name": "Mistral Small 3.2",
            "quality": 34.0,
            "cost": 0.15,
            "ttft": 0.51,
            "cost_pct": 4.4,
            "ttft_pct": 154.2,
            "score": 0.241
        },
        {
            "name": "Phi-4",
            "quality": 29.2,
            "cost": 0.22,
            "ttft": 1.24,
            "cost_pct": 6.4,
            "ttft_pct": 372.9,
            "score": 0.2563
        },
        {
            "name": "Llama 4 Scout",
            "quality": 28.2,
            "cost": 0.24,
            "ttft": 0.48,
            "cost_pct": 7.0,
            "ttft_pct": 144.8,
            "score": 0.2596
        },
        {
            "name": "Mistral Small 3.1",
            "quality": 27.9,
            "cost": 0.15,
            "ttft": 0.74,
            "cost_pct": 4.4,
            "ttft_pct": 223.9,
            "score": 0.2606
        },
        {
            "name": "Gemma 3 27B Instruct",
            "quality": 21.3,
            "cost": 0.0,
            "ttft": 0.41,
            "cost_pct": 0.0,
            "ttft_pct": 122.3,
            "score": 0.2817
        },
        {
            "name": "Gemma 3 12B Instruct",
            "quality": 17.2,
            "cost": 0.0,
            "ttft": 0.44,
            "cost_pct": 0.0,
            "ttft_pct": 132.7,
            "score": 0.2948
        }
    ],
    "coding_0_5": [
        {
            "name": "Grok 4.1 Fast (Reasoning)",
            "quality": 84.2,
            "cost": 0.28,
            "ttft": 0.63,
            "cost_pct": 8.0,
            "ttft_pct": 190.4,
            "score": 0.08
        },
        {
            "name": "gpt-oss-120B (high)",
            "quality": 83.6,
            "cost": 0.26,
            "ttft": 0.32,
            "cost_pct": 7.6,
            "ttft_pct": 94.9,
            "score": 0.082
        },
        {
            "name": "DeepSeek V3.2 Exp (Reasoning)",
            "quality": 71.5,
            "cost": 0.32,
            "ttft": 0.57,
            "cost_pct": 9.2,
            "ttft_pct": 172.1,
            "score": 0.1206
        },
        {
            "name": "Grok 3 mini Reasoning (high)",
            "quality": 67.3,
            "cost": 0.35,
            "ttft": 0.51,
            "cost_pct": 10.2,
            "ttft_pct": 152.8,
            "score": 0.1341
        },
        {
            "name": "GPT-5 nano (high)",
            "quality": 64.7,
            "cost": 0.14,
            "ttft": 0.27,
            "cost_pct": 4.0,
            "ttft_pct": 80.5,
            "score": 0.1426
        },
        {
            "name": "GLM-4.5-Air",
            "quality": 59.2,
            "cost": 0.42,
            "ttft": 0.69,
            "cost_pct": 12.4,
            "ttft_pct": 207.8,
            "score": 0.1616
        },
        {
            "name": "Gemini 2.5 Flash-Lite Preview (Sep '25) (Reasoning)",
            "quality": 56.7,
            "cost": 0.17,
            "ttft": 0.51,
            "cost_pct": 5.1,
            "ttft_pct": 154.0,
            "score": 0.1683
        },
        {
            "name": "Qwen3 4B 2507 (Reasoning)",
            "quality": 54.7,
            "cost": 0.0,
            "ttft": 0.91,
            "cost_pct": 0.0,
            "ttft_pct": 273.0,
            "score": 0.1983
        },
        {
            "name": "DeepSeek R1 0528 Qwen3 8B",
            "quality": 44.9,
            "cost": 0.07,
            "ttft": 0.77,
            "cost_pct": 2.0,
            "ttft_pct": 232.3,
            "score": 0.2058
        },
        {
            "name": "Llama 4 Maverick",
            "quality": 43.9,
            "cost": 0.42,
            "ttft": 0.47,
            "cost_pct": 12.3,
            "ttft_pct": 140.2,
            "score": 0.2091
        }
    ],
    "coding_1_0": [
        {
            "name": "GPT-5 mini (high)",
            "quality": 86.1,
            "cost": 0.69,
            "ttft": 0.34,
            "cost_pct": 20.0,
            "ttft_pct": 102.9,
            "score": 0.074
        },
        {
            "name": "Grok 4.1 Fast (Reasoning)",
            "quality": 84.2,
            "cost": 0.28,
            "ttft": 0.63,
            "cost_pct": 8.0,
            "ttft_pct": 190.4,
            "score": 0.08
        },
        {
            "name": "gpt-oss-120B (high)",
            "quality": 83.6,
            "cost": 0.26,
            "ttft": 0.32,
            "cost_pct": 7.6,
            "ttft_pct": 94.9,
            "score": 0.082
        },
        {
            "name": "DeepSeek V3.1 Terminus (Reasoning)",
            "quality": 77.5,
            "cost": 0.8,
            "ttft": 0.44,
            "cost_pct": 23.3,
            "ttft_pct": 132.7,
            "score": 0.1089
        },
        {
            "name": "DeepSeek V3.2 Exp (Reasoning)",
            "quality": 71.5,
            "cost": 0.32,
            "ttft": 0.57,
            "cost_pct": 9.2,
            "ttft_pct": 172.1,
            "score": 0.1206
        },
        {
            "name": "Grok 3 mini Reasoning (high)",
            "quality": 67.3,
            "cost": 0.35,
            "ttft": 0.51,
            "cost_pct": 10.2,
            "ttft_pct": 152.8,
            "score": 0.1341
        },
        {
            "name": "GPT-5 nano (high)",
            "quality": 64.7,
            "cost": 0.14,
            "ttft": 0.27,
            "cost_pct": 4.0,
            "ttft_pct": 80.5,
            "score": 0.1426
        },
        {
            "name": "Gemini 2.5 Flash Preview (Sep '25) (Reasoning)",
            "quality": 66.4,
            "cost": 0.85,
            "ttft": 0.63,
            "cost_pct": 24.7,
            "ttft_pct": 188.8,
            "score": 0.154
        },
        {
            "name": "GLM-4.5-Air",
            "quality": 59.2,
            "cost": 0.42,
            "ttft": 0.69,
            "cost_pct": 12.4,
            "ttft_pct": 207.8,
            "score": 0.1602
        },
        {
            "name": "Gemini 2.5 Flash-Lite Preview (Sep '25) (Reasoning)",
            "quality": 56.7,
            "cost": 0.17,
            "ttft": 0.51,
            "cost_pct": 5.1,
            "ttft_pct": 154.0,
            "score": 0.1683
        }
    ],
    "coding_2_0": [
        {
            "name": "GPT-5 mini (high)",
            "quality": 86.1,
            "cost": 0.69,
            "ttft": 0.34,
            "cost_pct": 20.0,
            "ttft_pct": 102.9,
            "score": 0.074
        },
        {
            "name": "Grok 4.1 Fast (Reasoning)",
            "quality": 84.2,
            "cost": 0.28,
            "ttft": 0.63,
            "cost_pct": 8.0,
            "ttft_pct": 190.4,
            "score": 0.08
        },
        {
            "name": "gpt-oss-120B (high)",
            "quality": 83.6,
            "cost": 0.26,
            "ttft": 0.32,
            "cost_pct": 7.6,
            "ttft_pct": 94.9,
            "score": 0.082
        },
        {
            "name": "DeepSeek V3.1 Terminus (Reasoning)",
            "quality": 77.5,
            "cost": 0.8,
            "ttft": 0.44,
            "cost_pct": 23.3,
            "ttft_pct": 132.7,
            "score": 0.1013
        },
        {
            "name": "DeepSeek V3.2 Exp (Reasoning)",
            "quality": 71.5,
            "cost": 0.32,
            "ttft": 0.57,
            "cost_pct": 9.2,
            "ttft_pct": 172.1,
            "score": 0.1206
        },
        {
            "name": "Grok 3 mini Reasoning (high)",
            "quality": 67.3,
            "cost": 0.35,
            "ttft": 0.51,
            "cost_pct": 10.2,
            "ttft_pct": 152.8,
            "score": 0.1341
        },
        {
            "name": "GPT-5 nano (high)",
            "quality": 64.7,
            "cost": 0.14,
            "ttft": 0.27,
            "cost_pct": 4.0,
            "ttft_pct": 80.5,
            "score": 0.1426
        },
        {
            "name": "Gemini 2.5 Flash Preview (Sep '25) (Reasoning)",
            "quality": 66.4,
            "cost": 0.85,
            "ttft": 0.63,
            "cost_pct": 24.7,
            "ttft_pct": 188.8,
            "score": 0.1432
        },
        {
            "name": "GLM-4.5-Air",
            "quality": 59.2,
            "cost": 0.42,
            "ttft": 0.69,
            "cost_pct": 12.4,
            "ttft_pct": 207.8,
            "score": 0.1602
        },
        {
            "name": "Gemini 2.5 Flash-Lite Preview (Sep '25) (Reasoning)",
            "quality": 56.7,
            "cost": 0.17,
            "ttft": 0.51,
            "cost_pct": 5.1,
            "ttft_pct": 154.0,
            "score": 0.1683
        }
    ],
    "coding_5_0": [
        {
            "name": "GPT-5 mini (high)",
            "quality": 86.1,
            "cost": 0.69,
            "ttft": 0.34,
            "cost_pct": 20.0,
            "ttft_pct": 102.9,
            "score": 0.074
        },
        {
            "name": "Grok 4.1 Fast (Reasoning)",
            "quality": 84.2,
            "cost": 0.28,
            "ttft": 0.63,
            "cost_pct": 8.0,
            "ttft_pct": 190.4,
            "score": 0.08
        },
        {
            "name": "gpt-oss-120B (high)",
            "quality": 83.6,
            "cost": 0.26,
            "ttft": 0.32,
            "cost_pct": 7.6,
            "ttft_pct": 94.9,
            "score": 0.082
        },
        {
            "name": "DeepSeek V3.1 Terminus (Reasoning)",
            "quality": 77.5,
            "cost": 0.8,
            "ttft": 0.44,
            "cost_pct": 23.3,
            "ttft_pct": 132.7,
            "score": 0.1013
        },
        {
            "name": "o4-mini (high)",
            "quality": 85.8,
            "cost": 1.93,
            "ttft": 0.26,
            "cost_pct": 56.0,
            "ttft_pct": 79.7,
            "score": 0.1118
        },
        {
            "name": "DeepSeek V3.2 Exp (Reasoning)",
            "quality": 71.5,
            "cost": 0.32,
            "ttft": 0.57,
            "cost_pct": 9.2,
            "ttft_pct": 172.1,
            "score": 0.1206
        },
        {
            "name": "GPT-5.1 (high)",
            "quality": 100.0,
            "cost": 3.44,
            "ttft": 0.33,
            "cost_pct": 100.0,
            "ttft_pct": 100.0,
            "score": 0.1257
        },
        {
            "name": "Grok 3 mini Reasoning (high)",
            "quality": 67.3,
            "cost": 0.35,
            "ttft": 0.51,
            "cost_pct": 10.2,
            "ttft_pct": 152.8,
            "score": 0.1341
        },
        {
            "name": "Gemini 2.5 Flash Preview (Sep '25) (Reasoning)",
            "quality": 66.4,
            "cost": 0.85,
            "ttft": 0.63,
            "cost_pct": 24.7,
            "ttft_pct": 188.8,
            "score": 0.1371
        },
        {
            "name": "GLM-4.6 (Reasoning)",
            "quality": 66.1,
            "cost": 1,
            "ttft": 1.69,
            "cost_pct": 29.1,
            "ttft_pct": 508.3,
            "score": 0.1381
        }
    ],
    "data_science_0_1": [
        {
            "name": "DeepSeek R1 0528 Qwen3 8B",
            "quality": 48.5,
            "cost": 0.07,
            "ttft": 0.77,
            "cost_pct": 2.0,
            "ttft_pct": 232.3,
            "score": 0.1908
        },
        {
            "name": "Qwen3 4B 2507 (Reasoning)",
            "quality": 55.0,
            "cost": 0.0,
            "ttft": 0.91,
            "cost_pct": 0.0,
            "ttft_pct": 273.0,
            "score": 0.2
        },
        {
            "name": "Gemma 3 27B Instruct",
            "quality": 26.8,
            "cost": 0.0,
            "ttft": 0.41,
            "cost_pct": 0.0,
            "ttft_pct": 122.3,
            "score": 0.2621
        },
        {
            "name": "Gemma 3 12B Instruct",
            "quality": 22.3,
            "cost": 0.0,
            "ttft": 0.44,
            "cost_pct": 0.0,
            "ttft_pct": 132.7,
            "score": 0.277
        },
        {
            "name": "Gemma 3 4B Instruct",
            "quality": 11.9,
            "cost": 0.0,
            "ttft": 0.29,
            "cost_pct": 0.0,
            "ttft_pct": 86.7,
            "score": 0.3111
        },
        {
            "name": "Phi-4 Mini Instruct",
            "quality": 10.7,
            "cost": 0.0,
            "ttft": 0.7,
            "cost_pct": 0.0,
            "ttft_pct": 211.8,
            "score": 0.3148
        },
        {
            "name": "Ministral 8B",
            "quality": 5.4,
            "cost": 0.1,
            "ttft": 0.72,
            "cost_pct": 2.9,
            "ttft_pct": 215.6,
            "score": 0.3323
        },
        {
            "name": "Ministral 3B",
            "quality": 1.5,
            "cost": 0.04,
            "ttft": 0.53,
            "cost_pct": 1.2,
            "ttft_pct": 158.6,
            "score": 0.3452
        }
    ],
    "data_science_0_25": [
        {
            "name": "GPT-5 nano (high)",
            "quality": 60.8,
            "cost": 0.14,
            "ttft": 0.27,
            "cost_pct": 4.0,
            "ttft_pct": 80.5,
            "score": 0.1505
        },
        {
            "name": "Qwen3 4B 2507 (Reasoning)",
            "quality": 55.0,
            "cost": 0.0,
            "ttft": 0.91,
            "cost_pct": 0.0,
            "ttft_pct": 273.0,
            "score": 0.1696
        },
        {
            "name": "Gemini 2.5 Flash-Lite Preview (Sep '25) (Reasoning)",
            "quality": 54.9,
            "cost": 0.17,
            "ttft": 0.51,
            "cost_pct": 5.1,
            "ttft_pct": 154.0,
            "score": 0.1698
        },
        {
            "name": "DeepSeek R1 0528 Qwen3 8B",
            "quality": 48.5,
            "cost": 0.07,
            "ttft": 0.77,
            "cost_pct": 2.0,
            "ttft_pct": 232.3,
            "score": 0.1908
        },
        {
            "name": "Mistral Small 3.2",
            "quality": 34.7,
            "cost": 0.15,
            "ttft": 0.51,
            "cost_pct": 4.4,
            "ttft_pct": 154.2,
            "score": 0.2362
        },
        {
            "name": "Llama 4 Scout",
            "quality": 30.2,
            "cost": 0.24,
            "ttft": 0.48,
            "cost_pct": 7.0,
            "ttft_pct": 144.8,
            "score": 0.2509
        },
        {
            "name": "Phi-4",
            "quality": 29.5,
            "cost": 0.22,
            "ttft": 1.24,
            "cost_pct": 6.4,
            "ttft_pct": 372.9,
            "score": 0.2533
        },
        {
            "name": "Gemma 3 27B Instruct",
            "quality": 26.8,
            "cost": 0.0,
            "ttft": 0.41,
            "cost_pct": 0.0,
            "ttft_pct": 122.3,
            "score": 0.2621
        },
        {
            "name": "Mistral Small 3.1",
            "quality": 22.9,
            "cost": 0.15,
            "ttft": 0.74,
            "cost_pct": 4.4,
            "ttft_pct": 223.9,
            "score": 0.2748
        },
        {
            "name": "Gemma 3 12B Instruct",
            "quality": 22.3,
            "cost": 0.0,
            "ttft": 0.44,
            "cost_pct": 0.0,
            "ttft_pct": 132.7,
            "score": 0.277
        }
    ],
    "data_science_0_5": [
        {
            "name": "Grok 4.1 Fast (Reasoning)",
            "quality": 83.6,
            "cost": 0.28,
            "ttft": 0.63,
            "cost_pct": 8.0,
            "ttft_pct": 190.4,
            "score": 0.0758
        },
        {
            "name": "gpt-oss-120B (high)",
            "quality": 81.7,
            "cost": 0.26,
            "ttft": 0.32,
            "cost_pct": 7.6,
            "ttft_pct": 94.9,
            "score": 0.0821
        },
        {
            "name": "DeepSeek V3.2 Exp (Reasoning)",
            "quality": 71.6,
            "cost": 0.32,
            "ttft": 0.57,
            "cost_pct": 9.2,
            "ttft_pct": 172.1,
            "score": 0.115
        },
        {
            "name": "Grok 3 mini Reasoning (high)",
            "quality": 71.3,
            "cost": 0.35,
            "ttft": 0.51,
            "cost_pct": 10.2,
            "ttft_pct": 152.8,
            "score": 0.1162
        },
        {
            "name": "GPT-5 nano (high)",
            "quality": 60.8,
            "cost": 0.14,
            "ttft": 0.27,
            "cost_pct": 4.0,
            "ttft_pct": 80.5,
            "score": 0.1505
        },
        {
            "name": "GLM-4.5-Air",
            "quality": 60.7,
            "cost": 0.42,
            "ttft": 0.69,
            "cost_pct": 12.4,
            "ttft_pct": 207.8,
            "score": 0.1573
        },
        {
            "name": "Gemini 2.5 Flash-Lite Preview (Sep '25) (Reasoning)",
            "quality": 54.9,
            "cost": 0.17,
            "ttft": 0.51,
            "cost_pct": 5.1,
            "ttft_pct": 154.0,
            "score": 0.1698
        },
        {
            "name": "Qwen3 4B 2507 (Reasoning)",
            "quality": 55.0,
            "cost": 0.0,
            "ttft": 0.91,
            "cost_pct": 0.0,
            "ttft_pct": 273.0,
            "score": 0.1851
        },
        {
            "name": "DeepSeek R1 0528 Qwen3 8B",
            "quality": 48.5,
            "cost": 0.07,
            "ttft": 0.77,
            "cost_pct": 2.0,
            "ttft_pct": 232.3,
            "score": 0.1908
        },
        {
            "name": "Llama 4 Maverick",
            "quality": 40.9,
            "cost": 0.42,
            "ttft": 0.47,
            "cost_pct": 12.3,
            "ttft_pct": 140.2,
            "score": 0.2159
        }
    ],
    "data_science_1_0": [
        {
            "name": "GPT-5 mini (high)",
            "quality": 85.8,
            "cost": 0.69,
            "ttft": 0.34,
            "cost_pct": 20.0,
            "ttft_pct": 102.9,
            "score": 0.0687
        },
        {
            "name": "Grok 4.1 Fast (Reasoning)",
            "quality": 83.6,
            "cost": 0.28,
            "ttft": 0.63,
            "cost_pct": 8.0,
            "ttft_pct": 190.4,
            "score": 0.0758
        },
        {
            "name": "gpt-oss-120B (high)",
            "quality": 81.7,
            "cost": 0.26,
            "ttft": 0.32,
            "cost_pct": 7.6,
            "ttft_pct": 94.9,
            "score": 0.0821
        },
        {
            "name": "DeepSeek V3.1 Terminus (Reasoning)",
            "quality": 77.5,
            "cost": 0.8,
            "ttft": 0.44,
            "cost_pct": 23.3,
            "ttft_pct": 132.7,
            "score": 0.1077
        },
        {
            "name": "DeepSeek V3.2 Exp (Reasoning)",
            "quality": 71.6,
            "cost": 0.32,
            "ttft": 0.57,
            "cost_pct": 9.2,
            "ttft_pct": 172.1,
            "score": 0.115
        },
        {
            "name": "Grok 3 mini Reasoning (high)",
            "quality": 71.3,
            "cost": 0.35,
            "ttft": 0.51,
            "cost_pct": 10.2,
            "ttft_pct": 152.8,
            "score": 0.1162
        },
        {
            "name": "GPT-5 nano (high)",
            "quality": 60.8,
            "cost": 0.14,
            "ttft": 0.27,
            "cost_pct": 4.0,
            "ttft_pct": 80.5,
            "score": 0.1505
        },
        {
            "name": "GLM-4.5-Air",
            "quality": 60.7,
            "cost": 0.42,
            "ttft": 0.69,
            "cost_pct": 12.4,
            "ttft_pct": 207.8,
            "score": 0.1508
        },
        {
            "name": "Gemini 2.5 Flash Preview (Sep '25) (Reasoning)",
            "quality": 65.1,
            "cost": 0.85,
            "ttft": 0.63,
            "cost_pct": 24.7,
            "ttft_pct": 188.8,
            "score": 0.1624
        },
        {
            "name": "Gemini 2.5 Flash-Lite Preview (Sep '25) (Reasoning)",
            "quality": 54.9,
            "cost": 0.17,
            "ttft": 0.51,
            "cost_pct": 5.1,
            "ttft_pct": 154.0,
            "score": 0.1698
        }
    ],
    "data_science_2_0": [
        {
            "name": "GPT-5 mini (high)",
            "quality": 85.8,
            "cost": 0.69,
            "ttft": 0.34,
            "cost_pct": 20.0,
            "ttft_pct": 102.9,
            "score": 0.0687
        },
        {
            "name": "Grok 4.1 Fast (Reasoning)",
            "quality": 83.6,
            "cost": 0.28,
            "ttft": 0.63,
            "cost_pct": 8.0,
            "ttft_pct": 190.4,
            "score": 0.0758
        },
        {
            "name": "gpt-oss-120B (high)",
            "quality": 81.7,
            "cost": 0.26,
            "ttft": 0.32,
            "cost_pct": 7.6,
            "ttft_pct": 94.9,
            "score": 0.0821
        },
        {
            "name": "DeepSeek V3.1 Terminus (Reasoning)",
            "quality": 77.5,
            "cost": 0.8,
            "ttft": 0.44,
            "cost_pct": 23.3,
            "ttft_pct": 132.7,
            "score": 0.1033
        },
        {
            "name": "DeepSeek V3.2 Exp (Reasoning)",
            "quality": 71.6,
            "cost": 0.32,
            "ttft": 0.57,
            "cost_pct": 9.2,
            "ttft_pct": 172.1,
            "score": 0.115
        },
        {
            "name": "Grok 3 mini Reasoning (high)",
            "quality": 71.3,
            "cost": 0.35,
            "ttft": 0.51,
            "cost_pct": 10.2,
            "ttft_pct": 152.8,
            "score": 0.1162
        },
        {
            "name": "GPT-5 nano (high)",
            "quality": 60.8,
            "cost": 0.14,
            "ttft": 0.27,
            "cost_pct": 4.0,
            "ttft_pct": 80.5,
            "score": 0.1505
        },
        {
            "name": "GLM-4.5-Air",
            "quality": 60.7,
            "cost": 0.42,
            "ttft": 0.69,
            "cost_pct": 12.4,
            "ttft_pct": 207.8,
            "score": 0.1508
        },
        {
            "name": "Gemini 2.5 Flash Preview (Sep '25) (Reasoning)",
            "quality": 65.1,
            "cost": 0.85,
            "ttft": 0.63,
            "cost_pct": 24.7,
            "ttft_pct": 188.8,
            "score": 0.1552
        },
        {
            "name": "Gemini 2.5 Flash-Lite Preview (Sep '25) (Reasoning)",
            "quality": 54.9,
            "cost": 0.17,
            "ttft": 0.51,
            "cost_pct": 5.1,
            "ttft_pct": 154.0,
            "score": 0.1698
        }
    ],
    "data_science_5_0": [
        {
            "name": "GPT-5 mini (high)",
            "quality": 85.8,
            "cost": 0.69,
            "ttft": 0.34,
            "cost_pct": 20.0,
            "ttft_pct": 102.9,
            "score": 0.0687
        },
        {
            "name": "Grok 4.1 Fast (Reasoning)",
            "quality": 83.6,
            "cost": 0.28,
            "ttft": 0.63,
            "cost_pct": 8.0,
            "ttft_pct": 190.4,
            "score": 0.0758
        },
        {
            "name": "gpt-oss-120B (high)",
            "quality": 81.7,
            "cost": 0.26,
            "ttft": 0.32,
            "cost_pct": 7.6,
            "ttft_pct": 94.9,
            "score": 0.0821
        },
        {
            "name": "DeepSeek V3.1 Terminus (Reasoning)",
            "quality": 77.5,
            "cost": 0.8,
            "ttft": 0.44,
            "cost_pct": 23.3,
            "ttft_pct": 132.7,
            "score": 0.0958
        },
        {
            "name": "DeepSeek V3.2 Exp (Reasoning)",
            "quality": 71.6,
            "cost": 0.32,
            "ttft": 0.57,
            "cost_pct": 9.2,
            "ttft_pct": 172.1,
            "score": 0.115
        },
        {
            "name": "Grok 3 mini Reasoning (high)",
            "quality": 71.3,
            "cost": 0.35,
            "ttft": 0.51,
            "cost_pct": 10.2,
            "ttft_pct": 152.8,
            "score": 0.1162
        },
        {
            "name": "GPT-5.1 (high)",
            "quality": 100.0,
            "cost": 3.44,
            "ttft": 0.33,
            "cost_pct": 100.0,
            "ttft_pct": 100.0,
            "score": 0.1283
        },
        {
            "name": "GLM-4.6 (Reasoning)",
            "quality": 67.5,
            "cost": 1,
            "ttft": 1.69,
            "cost_pct": 29.1,
            "ttft_pct": 508.3,
            "score": 0.1285
        },
        {
            "name": "o4-mini (high)",
            "quality": 82.2,
            "cost": 1.93,
            "ttft": 0.26,
            "cost_pct": 56.0,
            "ttft_pct": 79.7,
            "score": 0.1291
        },
        {
            "name": "Gemini 2.5 Flash Preview (Sep '25) (Reasoning)",
            "quality": 65.1,
            "cost": 0.85,
            "ttft": 0.63,
            "cost_pct": 24.7,
            "ttft_pct": 188.8,
            "score": 0.1363
        }
    ],
    "creative_0_1": [
        {
            "name": "DeepSeek R1 0528 Qwen3 8B",
            "quality": 48.1,
            "cost": 0.07,
            "ttft": 0.77,
            "cost_pct": 0.7,
            "ttft_pct": 45.0,
            "score": 0.1992
        },
        {
            "name": "Qwen3 4B 2507 (Reasoning)",
            "quality": 55.2,
            "cost": 0.0,
            "ttft": 0.91,
            "cost_pct": 0.0,
            "ttft_pct": 52.8,
            "score": 0.2
        },
        {
            "name": "Gemma 3 27B Instruct",
            "quality": 33.6,
            "cost": 0.0,
            "ttft": 0.41,
            "cost_pct": 0.0,
            "ttft_pct": 23.7,
            "score": 0.2448
        },
        {
            "name": "Gemma 3 12B Instruct",
            "quality": 32.4,
            "cost": 0.0,
            "ttft": 0.44,
            "cost_pct": 0.0,
            "ttft_pct": 25.7,
            "score": 0.2485
        },
        {
            "name": "Phi-4 Mini Instruct",
            "quality": 23.6,
            "cost": 0.0,
            "ttft": 0.7,
            "cost_pct": 0.0,
            "ttft_pct": 41.0,
            "score": 0.276
        },
        {
            "name": "Gemma 3 4B Instruct",
            "quality": 19.1,
            "cost": 0.0,
            "ttft": 0.29,
            "cost_pct": 0.0,
            "ttft_pct": 16.8,
            "score": 0.3
        },
        {
            "name": "Ministral 8B",
            "quality": 13.8,
            "cost": 0.1,
            "ttft": 0.72,
            "cost_pct": 1.0,
            "ttft_pct": 41.7,
            "score": 0.3066
        },
        {
            "name": "Ministral 3B",
            "quality": 11.6,
            "cost": 0.04,
            "ttft": 0.53,
            "cost_pct": 0.4,
            "ttft_pct": 30.7,
            "score": 0.3138
        }
    ],
    "creative_0_25": [
        {
            "name": "Gemini 2.5 Flash-Lite Preview (Sep '25) (Reasoning)",
            "quality": 60.3,
            "cost": 0.17,
            "ttft": 0.51,
            "cost_pct": 1.7,
            "ttft_pct": 29.8,
            "score": 0.1611
        },
        {
            "name": "GPT-5 nano (high)",
            "quality": 57.5,
            "cost": 0.14,
            "ttft": 0.27,
            "cost_pct": 1.4,
            "ttft_pct": 15.6,
            "score": 0.1698
        },
        {
            "name": "Qwen3 4B 2507 (Reasoning)",
            "quality": 55.2,
            "cost": 0.0,
            "ttft": 0.91,
            "cost_pct": 0.0,
            "ttft_pct": 52.8,
            "score": 0.1772
        },
        {
            "name": "DeepSeek R1 0528 Qwen3 8B",
            "quality": 48.1,
            "cost": 0.07,
            "ttft": 0.77,
            "cost_pct": 0.7,
            "ttft_pct": 45.0,
            "score": 0.1992
        },
        {
            "name": "Mistral Small 3.2",
            "quality": 41.6,
            "cost": 0.15,
            "ttft": 0.51,
            "cost_pct": 1.5,
            "ttft_pct": 29.8,
            "score": 0.2197
        },
        {
            "name": "Llama 4 Scout",
            "quality": 41.6,
            "cost": 0.24,
            "ttft": 0.48,
            "cost_pct": 2.4,
            "ttft_pct": 28.0,
            "score": 0.2198
        },
        {
            "name": "Phi-4",
            "quality": 40.5,
            "cost": 0.22,
            "ttft": 1.24,
            "cost_pct": 2.2,
            "ttft_pct": 72.2,
            "score": 0.2232
        },
        {
            "name": "Mistral Small 3.1",
            "quality": 38.4,
            "cost": 0.15,
            "ttft": 0.74,
            "cost_pct": 1.5,
            "ttft_pct": 43.3,
            "score": 0.2296
        },
        {
            "name": "Gemma 3 27B Instruct",
            "quality": 33.6,
            "cost": 0.0,
            "ttft": 0.41,
            "cost_pct": 0.0,
            "ttft_pct": 23.7,
            "score": 0.2448
        },
        {
            "name": "Gemma 3 12B Instruct",
            "quality": 32.4,
            "cost": 0.0,
            "ttft": 0.44,
            "cost_pct": 0.0,
            "ttft_pct": 25.7,
            "score": 0.2634
        }
    ],
    "creative_0_5": [
        {
            "name": "Grok 4.1 Fast (Reasoning)",
            "quality": 82.6,
            "cost": 0.28,
            "ttft": 0.63,
            "cost_pct": 2.8,
            "ttft_pct": 36.9,
            "score": 0.0913
        },
        {
            "name": "DeepSeek V3.2 Exp (Reasoning)",
            "quality": 76.2,
            "cost": 0.32,
            "ttft": 0.57,
            "cost_pct": 3.1,
            "ttft_pct": 33.3,
            "score": 0.1112
        },
        {
            "name": "Grok 3 mini Reasoning (high)",
            "quality": 70.9,
            "cost": 0.35,
            "ttft": 0.51,
            "cost_pct": 3.5,
            "ttft_pct": 29.6,
            "score": 0.1279
        },
        {
            "name": "gpt-oss-120B (high)",
            "quality": 68.5,
            "cost": 0.26,
            "ttft": 0.32,
            "cost_pct": 2.6,
            "ttft_pct": 18.4,
            "score": 0.1355
        },
        {
            "name": "Gemini 2.5 Flash-Lite Preview (Sep '25) (Reasoning)",
            "quality": 60.3,
            "cost": 0.17,
            "ttft": 0.51,
            "cost_pct": 1.7,
            "ttft_pct": 29.8,
            "score": 0.1611
        },
        {
            "name": "GLM-4.5-Air",
            "quality": 59.1,
            "cost": 0.42,
            "ttft": 0.69,
            "cost_pct": 4.2,
            "ttft_pct": 40.2,
            "score": 0.165
        },
        {
            "name": "GPT-5 nano (high)",
            "quality": 57.5,
            "cost": 0.14,
            "ttft": 0.27,
            "cost_pct": 1.4,
            "ttft_pct": 15.6,
            "score": 0.1698
        },
        {
            "name": "Qwen3 4B 2507 (Reasoning)",
            "quality": 55.2,
            "cost": 0.0,
            "ttft": 0.91,
            "cost_pct": 0.0,
            "ttft_pct": 52.8,
            "score": 0.1843
        },
        {
            "name": "Llama 4 Maverick",
            "quality": 50.4,
            "cost": 0.42,
            "ttft": 0.47,
            "cost_pct": 4.2,
            "ttft_pct": 27.1,
            "score": 0.192
        },
        {
            "name": "DeepSeek R1 0528 Qwen3 8B",
            "quality": 48.1,
            "cost": 0.07,
            "ttft": 0.77,
            "cost_pct": 0.7,
            "ttft_pct": 45.0,
            "score": 0.1992
        }
    ],
    "creative_1_0": [
        {
            "name": "Grok 4.1 Fast (Reasoning)",
            "quality": 82.6,
            "cost": 0.28,
            "ttft": 0.63,
            "cost_pct": 2.8,
            "ttft_pct": 36.9,
            "score": 0.0913
        },
        {
            "name": "GPT-5 mini (high)",
            "quality": 80.8,
            "cost": 0.69,
            "ttft": 0.34,
            "cost_pct": 6.9,
            "ttft_pct": 19.9,
            "score": 0.0969
        },
        {
            "name": "DeepSeek V3.1 Terminus (Reasoning)",
            "quality": 77.6,
            "cost": 0.8,
            "ttft": 0.44,
            "cost_pct": 8.0,
            "ttft_pct": 25.7,
            "score": 0.107
        },
        {
            "name": "DeepSeek V3.2 Exp (Reasoning)",
            "quality": 76.2,
            "cost": 0.32,
            "ttft": 0.57,
            "cost_pct": 3.1,
            "ttft_pct": 33.3,
            "score": 0.1112
        },
        {
            "name": "Grok 3 mini Reasoning (high)",
            "quality": 70.9,
            "cost": 0.35,
            "ttft": 0.51,
            "cost_pct": 3.5,
            "ttft_pct": 29.6,
            "score": 0.1279
        },
        {
            "name": "gpt-oss-120B (high)",
            "quality": 68.5,
            "cost": 0.26,
            "ttft": 0.32,
            "cost_pct": 2.6,
            "ttft_pct": 18.4,
            "score": 0.1355
        },
        {
            "name": "Gemini 2.5 Flash Preview (Sep '25) (Reasoning)",
            "quality": 70.1,
            "cost": 0.85,
            "ttft": 0.63,
            "cost_pct": 8.5,
            "ttft_pct": 36.5,
            "score": 0.1391
        },
        {
            "name": "Gemini 2.5 Flash-Lite Preview (Sep '25) (Reasoning)",
            "quality": 60.3,
            "cost": 0.17,
            "ttft": 0.51,
            "cost_pct": 1.7,
            "ttft_pct": 29.8,
            "score": 0.1611
        },
        {
            "name": "GLM-4.5-Air",
            "quality": 59.1,
            "cost": 0.42,
            "ttft": 0.69,
            "cost_pct": 4.2,
            "ttft_pct": 40.2,
            "score": 0.165
        },
        {
            "name": "GPT-5 nano (high)",
            "quality": 57.5,
            "cost": 0.14,
            "ttft": 0.27,
            "cost_pct": 1.4,
            "ttft_pct": 15.6,
            "score": 0.1698
        }
    ],
    "creative_2_0": [
        {
            "name": "Grok 4.1 Fast (Reasoning)",
            "quality": 82.6,
            "cost": 0.28,
            "ttft": 0.63,
            "cost_pct": 2.8,
            "ttft_pct": 36.9,
            "score": 0.0913
        },
        {
            "name": "GPT-5 mini (high)",
            "quality": 80.8,
            "cost": 0.69,
            "ttft": 0.34,
            "cost_pct": 6.9,
            "ttft_pct": 19.9,
            "score": 0.0969
        },
        {
            "name": "DeepSeek V3.1 Terminus (Reasoning)",
            "quality": 77.6,
            "cost": 0.8,
            "ttft": 0.44,
            "cost_pct": 8.0,
            "ttft_pct": 25.7,
            "score": 0.107
        },
        {
            "name": "DeepSeek V3.2 Exp (Reasoning)",
            "quality": 76.2,
            "cost": 0.32,
            "ttft": 0.57,
            "cost_pct": 3.1,
            "ttft_pct": 33.3,
            "score": 0.1112
        },
        {
            "name": "Grok 3 mini Reasoning (high)",
            "quality": 70.9,
            "cost": 0.35,
            "ttft": 0.51,
            "cost_pct": 3.5,
            "ttft_pct": 29.6,
            "score": 0.1279
        },
        {
            "name": "Gemini 2.5 Flash Preview (Sep '25) (Reasoning)",
            "quality": 70.1,
            "cost": 0.85,
            "ttft": 0.63,
            "cost_pct": 8.5,
            "ttft_pct": 36.5,
            "score": 0.1333
        },
        {
            "name": "gpt-oss-120B (high)",
            "quality": 68.5,
            "cost": 0.26,
            "ttft": 0.32,
            "cost_pct": 2.6,
            "ttft_pct": 18.4,
            "score": 0.1355
        },
        {
            "name": "Gemini 2.5 Flash-Lite Preview (Sep '25) (Reasoning)",
            "quality": 60.3,
            "cost": 0.17,
            "ttft": 0.51,
            "cost_pct": 1.7,
            "ttft_pct": 29.8,
            "score": 0.1611
        },
        {
            "name": "GLM-4.5-Air",
            "quality": 59.1,
            "cost": 0.42,
            "ttft": 0.69,
            "cost_pct": 4.2,
            "ttft_pct": 40.2,
            "score": 0.165
        },
        {
            "name": "GPT-5 nano (high)",
            "quality": 57.5,
            "cost": 0.14,
            "ttft": 0.27,
            "cost_pct": 1.4,
            "ttft_pct": 15.6,
            "score": 0.1698
        }
    ],
    "creative_5_0": [
        {
            "name": "Grok 4.1 Fast (Reasoning)",
            "quality": 82.6,
            "cost": 0.28,
            "ttft": 0.63,
            "cost_pct": 2.8,
            "ttft_pct": 36.9,
            "score": 0.0913
        },
        {
            "name": "GPT-5 mini (high)",
            "quality": 80.8,
            "cost": 0.69,
            "ttft": 0.34,
            "cost_pct": 6.9,
            "ttft_pct": 19.9,
            "score": 0.0969
        },
        {
            "name": "DeepSeek V3.1 Terminus (Reasoning)",
            "quality": 77.6,
            "cost": 0.8,
            "ttft": 0.44,
            "cost_pct": 8.0,
            "ttft_pct": 25.7,
            "score": 0.107
        },
        {
            "name": "DeepSeek V3.2 Exp (Reasoning)",
            "quality": 76.2,
            "cost": 0.32,
            "ttft": 0.57,
            "cost_pct": 3.1,
            "ttft_pct": 33.3,
            "score": 0.1112
        },
        {
            "name": "GPT-5.1 (high)",
            "quality": 97.2,
            "cost": 3.44,
            "ttft": 0.33,
            "cost_pct": 34.4,
            "ttft_pct": 19.4,
            "score": 0.1263
        },
        {
            "name": "Grok 3 mini Reasoning (high)",
            "quality": 70.9,
            "cost": 0.35,
            "ttft": 0.51,
            "cost_pct": 3.5,
            "ttft_pct": 29.6,
            "score": 0.1279
        },
        {
            "name": "Gemini 2.5 Flash Preview (Sep '25) (Reasoning)",
            "quality": 70.1,
            "cost": 0.85,
            "ttft": 0.63,
            "cost_pct": 8.5,
            "ttft_pct": 36.5,
            "score": 0.1304
        },
        {
            "name": "gpt-oss-120B (high)",
            "quality": 68.5,
            "cost": 0.26,
            "ttft": 0.32,
            "cost_pct": 2.6,
            "ttft_pct": 18.4,
            "score": 0.1355
        },
        {
            "name": "GLM-4.6 (Reasoning)",
            "quality": 68.2,
            "cost": 1,
            "ttft": 1.69,
            "cost_pct": 10.0,
            "ttft_pct": 98.4,
            "score": 0.1362
        },
        {
            "name": "Gemini 2.5 Pro",
            "quality": 84.8,
            "cost": 3.44,
            "ttft": 0.94,
            "cost_pct": 34.4,
            "ttft_pct": 55.0,
            "score": 0.1432
        }
    ],
    "general_0_1": [
        {
            "name": "DeepSeek R1 0528 Qwen3 8B",
            "quality": 60.3,
            "cost": 0.07,
            "ttft": 0.77,
            "cost_pct": 2.0,
            "ttft_pct": 232.3,
            "score": 0.1821
        },
        {
            "name": "Qwen3 4B 2507 (Reasoning)",
            "quality": 64.6,
            "cost": 0.0,
            "ttft": 0.91,
            "cost_pct": 0.0,
            "ttft_pct": 273.0,
            "score": 0.2
        },
        {
            "name": "Gemma 3 12B Instruct",
            "quality": 42.6,
            "cost": 0.0,
            "ttft": 0.44,
            "cost_pct": 0.0,
            "ttft_pct": 132.7,
            "score": 0.2314
        },
        {
            "name": "Gemma 3 27B Instruct",
            "quality": 38.2,
            "cost": 0.0,
            "ttft": 0.41,
            "cost_pct": 0.0,
            "ttft_pct": 122.3,
            "score": 0.2438
        },
        {
            "name": "Phi-4 Mini Instruct",
            "quality": 35.8,
            "cost": 0.0,
            "ttft": 0.7,
            "cost_pct": 0.0,
            "ttft_pct": 211.8,
            "score": 0.2605
        },
        {
            "name": "Ministral 8B",
            "quality": 20.8,
            "cost": 0.1,
            "ttft": 0.72,
            "cost_pct": 2.9,
            "ttft_pct": 215.6,
            "score": 0.292
        },
        {
            "name": "Ministral 3B",
            "quality": 18.4,
            "cost": 0.04,
            "ttft": 0.53,
            "cost_pct": 1.2,
            "ttft_pct": 158.6,
            "score": 0.2988
        },
        {
            "name": "Gemma 3 4B Instruct",
            "quality": 27.6,
            "cost": 0.0,
            "ttft": 0.29,
            "cost_pct": 0.0,
            "ttft_pct": 86.7,
            "score": 0.3
        }
    ],
    "general_0_25": [
        {
            "name": "Gemini 2.5 Flash-Lite Preview (Sep '25) (Reasoning)",
            "quality": 72.0,
            "cost": 0.17,
            "ttft": 0.51,
            "cost_pct": 5.1,
            "ttft_pct": 154.0,
            "score": 0.1496
        },
        {
            "name": "GPT-5 nano (high)",
            "quality": 65.8,
            "cost": 0.14,
            "ttft": 0.27,
            "cost_pct": 4.0,
            "ttft_pct": 80.5,
            "score": 0.1668
        },
        {
            "name": "Qwen3 4B 2507 (Reasoning)",
            "quality": 64.6,
            "cost": 0.0,
            "ttft": 0.91,
            "cost_pct": 0.0,
            "ttft_pct": 273.0,
            "score": 0.1702
        },
        {
            "name": "DeepSeek R1 0528 Qwen3 8B",
            "quality": 60.3,
            "cost": 0.07,
            "ttft": 0.77,
            "cost_pct": 2.0,
            "ttft_pct": 232.3,
            "score": 0.1821
        },
        {
            "name": "Mistral Small 3.2",
            "quality": 51.3,
            "cost": 0.15,
            "ttft": 0.51,
            "cost_pct": 4.4,
            "ttft_pct": 154.2,
            "score": 0.2072
        },
        {
            "name": "Phi-4",
            "quality": 50.7,
            "cost": 0.22,
            "ttft": 1.24,
            "cost_pct": 6.4,
            "ttft_pct": 372.9,
            "score": 0.2088
        },
        {
            "name": "Mistral Small 3.1",
            "quality": 46.0,
            "cost": 0.15,
            "ttft": 0.74,
            "cost_pct": 4.4,
            "ttft_pct": 223.9,
            "score": 0.2219
        },
        {
            "name": "Llama 4 Scout",
            "quality": 44.2,
            "cost": 0.24,
            "ttft": 0.48,
            "cost_pct": 7.0,
            "ttft_pct": 144.8,
            "score": 0.2269
        },
        {
            "name": "Gemma 3 12B Instruct",
            "quality": 42.6,
            "cost": 0.0,
            "ttft": 0.44,
            "cost_pct": 0.0,
            "ttft_pct": 132.7,
            "score": 0.237
        },
        {
            "name": "Gemma 3 27B Instruct",
            "quality": 38.2,
            "cost": 0.0,
            "ttft": 0.41,
            "cost_pct": 0.0,
            "ttft_pct": 122.3,
            "score": 0.2613
        }
    ],
    "general_0_5": [
        {
            "name": "DeepSeek V3.2 Exp (Reasoning)",
            "quality": 83.8,
            "cost": 0.32,
            "ttft": 0.57,
            "cost_pct": 9.2,
            "ttft_pct": 172.1,
            "score": 0.1166
        },
        {
            "name": "Grok 4.1 Fast (Reasoning)",
            "quality": 81.0,
            "cost": 0.28,
            "ttft": 0.63,
            "cost_pct": 8.0,
            "ttft_pct": 190.4,
            "score": 0.1244
        },
        {
            "name": "Grok 3 mini Reasoning (high)",
            "quality": 80.2,
            "cost": 0.35,
            "ttft": 0.51,
            "cost_pct": 10.2,
            "ttft_pct": 152.8,
            "score": 0.1267
        },
        {
            "name": "gpt-oss-120B (high)",
            "quality": 76.6,
            "cost": 0.26,
            "ttft": 0.32,
            "cost_pct": 7.6,
            "ttft_pct": 94.9,
            "score": 0.1367
        },
        {
            "name": "Gemini 2.5 Flash-Lite Preview (Sep '25) (Reasoning)",
            "quality": 72.0,
            "cost": 0.17,
            "ttft": 0.51,
            "cost_pct": 5.1,
            "ttft_pct": 154.0,
            "score": 0.1496
        },
        {
            "name": "GLM-4.5-Air",
            "quality": 66.4,
            "cost": 0.42,
            "ttft": 0.69,
            "cost_pct": 12.4,
            "ttft_pct": 207.8,
            "score": 0.1651
        },
        {
            "name": "GPT-5 nano (high)",
            "quality": 65.8,
            "cost": 0.14,
            "ttft": 0.27,
            "cost_pct": 4.0,
            "ttft_pct": 80.5,
            "score": 0.1668
        },
        {
            "name": "Qwen3 4B 2507 (Reasoning)",
            "quality": 64.6,
            "cost": 0.0,
            "ttft": 0.91,
            "cost_pct": 0.0,
            "ttft_pct": 273.0,
            "score": 0.1702
        },
        {
            "name": "DeepSeek R1 0528 Qwen3 8B",
            "quality": 60.3,
            "cost": 0.07,
            "ttft": 0.77,
            "cost_pct": 2.0,
            "ttft_pct": 232.3,
            "score": 0.1821
        },
        {
            "name": "Llama 4 Maverick",
            "quality": 53.9,
            "cost": 0.42,
            "ttft": 0.47,
            "cost_pct": 12.3,
            "ttft_pct": 140.2,
            "score": 0.1999
        }
    ],
    "general_1_0": [
        {
            "name": "DeepSeek V3.1 Terminus (Reasoning)",
            "quality": 87.0,
            "cost": 0.8,
            "ttft": 0.44,
            "cost_pct": 23.3,
            "ttft_pct": 132.7,
            "score": 0.1079
        },
        {
            "name": "GPT-5 mini (high)",
            "quality": 85.2,
            "cost": 0.69,
            "ttft": 0.34,
            "cost_pct": 20.0,
            "ttft_pct": 102.9,
            "score": 0.1128
        },
        {
            "name": "DeepSeek V3.2 Exp (Reasoning)",
            "quality": 83.8,
            "cost": 0.32,
            "ttft": 0.57,
            "cost_pct": 9.2,
            "ttft_pct": 172.1,
            "score": 0.1166
        },
        {
            "name": "Grok 4.1 Fast (Reasoning)",
            "quality": 81.0,
            "cost": 0.28,
            "ttft": 0.63,
            "cost_pct": 8.0,
            "ttft_pct": 190.4,
            "score": 0.1244
        },
        {
            "name": "Grok 3 mini Reasoning (high)",
            "quality": 80.2,
            "cost": 0.35,
            "ttft": 0.51,
            "cost_pct": 10.2,
            "ttft_pct": 152.8,
            "score": 0.1267
        },
        {
            "name": "gpt-oss-120B (high)",
            "quality": 76.6,
            "cost": 0.26,
            "ttft": 0.32,
            "cost_pct": 7.6,
            "ttft_pct": 94.9,
            "score": 0.1367
        },
        {
            "name": "Gemini 2.5 Flash Preview (Sep '25) (Reasoning)",
            "quality": 74.4,
            "cost": 0.85,
            "ttft": 0.63,
            "cost_pct": 24.7,
            "ttft_pct": 188.8,
            "score": 0.143
        },
        {
            "name": "Gemini 2.5 Flash-Lite Preview (Sep '25) (Reasoning)",
            "quality": 72.0,
            "cost": 0.17,
            "ttft": 0.51,
            "cost_pct": 5.1,
            "ttft_pct": 154.0,
            "score": 0.1496
        },
        {
            "name": "GLM-4.5-Air",
            "quality": 66.4,
            "cost": 0.42,
            "ttft": 0.69,
            "cost_pct": 12.4,
            "ttft_pct": 207.8,
            "score": 0.1651
        },
        {
            "name": "GPT-5 nano (high)",
            "quality": 65.8,
            "cost": 0.14,
            "ttft": 0.27,
            "cost_pct": 4.0,
            "ttft_pct": 80.5,
            "score": 0.1668
        }
    ],
    "general_2_0": [
        {
            "name": "DeepSeek V3.1 Terminus (Reasoning)",
            "quality": 87.0,
            "cost": 0.8,
            "ttft": 0.44,
            "cost_pct": 23.3,
            "ttft_pct": 132.7,
            "score": 0.1079
        },
        {
            "name": "GPT-5 mini (high)",
            "quality": 85.2,
            "cost": 0.69,
            "ttft": 0.34,
            "cost_pct": 20.0,
            "ttft_pct": 102.9,
            "score": 0.1128
        },
        {
            "name": "DeepSeek V3.2 Exp (Reasoning)",
            "quality": 83.8,
            "cost": 0.32,
            "ttft": 0.57,
            "cost_pct": 9.2,
            "ttft_pct": 172.1,
            "score": 0.1166
        },
        {
            "name": "Grok 4.1 Fast (Reasoning)",
            "quality": 81.0,
            "cost": 0.28,
            "ttft": 0.63,
            "cost_pct": 8.0,
            "ttft_pct": 190.4,
            "score": 0.1244
        },
        {
            "name": "Grok 3 mini Reasoning (high)",
            "quality": 80.2,
            "cost": 0.35,
            "ttft": 0.51,
            "cost_pct": 10.2,
            "ttft_pct": 152.8,
            "score": 0.1267
        },
        {
            "name": "gpt-oss-120B (high)",
            "quality": 76.6,
            "cost": 0.26,
            "ttft": 0.32,
            "cost_pct": 7.6,
            "ttft_pct": 94.9,
            "score": 0.1367
        },
        {
            "name": "Gemini 2.5 Flash Preview (Sep '25) (Reasoning)",
            "quality": 74.4,
            "cost": 0.85,
            "ttft": 0.63,
            "cost_pct": 24.7,
            "ttft_pct": 188.8,
            "score": 0.143
        },
        {
            "name": "Gemini 2.5 Flash-Lite Preview (Sep '25) (Reasoning)",
            "quality": 72.0,
            "cost": 0.17,
            "ttft": 0.51,
            "cost_pct": 5.1,
            "ttft_pct": 154.0,
            "score": 0.1496
        },
        {
            "name": "GLM-4.5-Air",
            "quality": 66.4,
            "cost": 0.42,
            "ttft": 0.69,
            "cost_pct": 12.4,
            "ttft_pct": 207.8,
            "score": 0.1651
        },
        {
            "name": "GPT-5 nano (high)",
            "quality": 65.8,
            "cost": 0.14,
            "ttft": 0.27,
            "cost_pct": 4.0,
            "ttft_pct": 80.5,
            "score": 0.1668
        }
    ],
    "general_5_0": [
        {
            "name": "DeepSeek V3.1 Terminus (Reasoning)",
            "quality": 87.0,
            "cost": 0.8,
            "ttft": 0.44,
            "cost_pct": 23.3,
            "ttft_pct": 132.7,
            "score": 0.1079
        },
        {
            "name": "GPT-5 mini (high)",
            "quality": 85.2,
            "cost": 0.69,
            "ttft": 0.34,
            "cost_pct": 20.0,
            "ttft_pct": 102.9,
            "score": 0.1128
        },
        {
            "name": "DeepSeek V3.2 Exp (Reasoning)",
            "quality": 83.8,
            "cost": 0.32,
            "ttft": 0.57,
            "cost_pct": 9.2,
            "ttft_pct": 172.1,
            "score": 0.1166
        },
        {
            "name": "Grok 4.1 Fast (Reasoning)",
            "quality": 81.0,
            "cost": 0.28,
            "ttft": 0.63,
            "cost_pct": 8.0,
            "ttft_pct": 190.4,
            "score": 0.1244
        },
        {
            "name": "GPT-5.1 (high)",
            "quality": 100.0,
            "cost": 3.44,
            "ttft": 0.33,
            "cost_pct": 100.0,
            "ttft_pct": 100.0,
            "score": 0.125
        },
        {
            "name": "Gemini 2.5 Pro",
            "quality": 88.1,
            "cost": 3.44,
            "ttft": 0.94,
            "cost_pct": 100.0,
            "ttft_pct": 284.2,
            "score": 0.1259
        },
        {
            "name": "Grok 3 mini Reasoning (high)",
            "quality": 80.2,
            "cost": 0.35,
            "ttft": 0.51,
            "cost_pct": 10.2,
            "ttft_pct": 152.8,
            "score": 0.1267
        },
        {
            "name": "gpt-oss-120B (high)",
            "quality": 76.6,
            "cost": 0.26,
            "ttft": 0.32,
            "cost_pct": 7.6,
            "ttft_pct": 94.9,
            "score": 0.1367
        },
        {
            "name": "o4-mini (high)",
            "quality": 75.7,
            "cost": 1.93,
            "ttft": 0.26,
            "cost_pct": 56.0,
            "ttft_pct": 79.7,
            "score": 0.1407
        },
        {
            "name": "Gemini 2.5 Flash Preview (Sep '25) (Reasoning)",
            "quality": 74.4,
            "cost": 0.85,
            "ttft": 0.63,
            "cost_pct": 24.7,
            "ttft_pct": 188.8,
            "score": 0.143
        }
    ],
    "qa_0_1": [
        {
            "name": "DeepSeek R1 0528 Qwen3 8B",
            "quality": 69.6,
            "cost": 0.07,
            "ttft": 0.77,
            "cost_pct": 0.7,
            "ttft_pct": 45.0,
            "score": 0.1732
        },
        {
            "name": "Qwen3 4B 2507 (Reasoning)",
            "quality": 72.0,
            "cost": 0.0,
            "ttft": 0.91,
            "cost_pct": 0.0,
            "ttft_pct": 52.8,
            "score": 0.2
        },
        {
            "name": "Gemma 3 12B Instruct",
            "quality": 57.7,
            "cost": 0.0,
            "ttft": 0.44,
            "cost_pct": 0.0,
            "ttft_pct": 25.7,
            "score": 0.2032
        },
        {
            "name": "Phi-4 Mini Instruct",
            "quality": 53.1,
            "cost": 0.0,
            "ttft": 0.7,
            "cost_pct": 0.0,
            "ttft_pct": 41.0,
            "score": 0.2149
        },
        {
            "name": "Gemma 3 27B Instruct",
            "quality": 47.6,
            "cost": 0.0,
            "ttft": 0.41,
            "cost_pct": 0.0,
            "ttft_pct": 23.7,
            "score": 0.2559
        },
        {
            "name": "Ministral 8B",
            "quality": 31.1,
            "cost": 0.1,
            "ttft": 0.72,
            "cost_pct": 1.0,
            "ttft_pct": 41.7,
            "score": 0.2709
        },
        {
            "name": "Ministral 3B",
            "quality": 29.4,
            "cost": 0.04,
            "ttft": 0.53,
            "cost_pct": 0.4,
            "ttft_pct": 30.7,
            "score": 0.2753
        },
        {
            "name": "Gemma 3 4B Instruct",
            "quality": 40.0,
            "cost": 0.0,
            "ttft": 0.29,
            "cost_pct": 0.0,
            "ttft_pct": 16.8,
            "score": 0.3
        }
    ],
    "qa_0_25": [
        {
            "name": "Gemini 2.5 Flash-Lite Preview (Sep '25) (Reasoning)",
            "quality": 85.1,
            "cost": 0.17,
            "ttft": 0.51,
            "cost_pct": 1.7,
            "ttft_pct": 29.8,
            "score": 0.1338
        },
        {
            "name": "Qwen3 4B 2507 (Reasoning)",
            "quality": 72.0,
            "cost": 0.0,
            "ttft": 0.91,
            "cost_pct": 0.0,
            "ttft_pct": 52.8,
            "score": 0.167
        },
        {
            "name": "DeepSeek R1 0528 Qwen3 8B",
            "quality": 69.6,
            "cost": 0.07,
            "ttft": 0.77,
            "cost_pct": 0.7,
            "ttft_pct": 45.0,
            "score": 0.1732
        },
        {
            "name": "GPT-5 nano (high)",
            "quality": 69.5,
            "cost": 0.14,
            "ttft": 0.27,
            "cost_pct": 1.4,
            "ttft_pct": 15.6,
            "score": 0.1735
        },
        {
            "name": "Mistral Small 3.2",
            "quality": 63.7,
            "cost": 0.15,
            "ttft": 0.51,
            "cost_pct": 1.5,
            "ttft_pct": 29.8,
            "score": 0.1882
        },
        {
            "name": "Mistral Small 3.1",
            "quality": 60.0,
            "cost": 0.15,
            "ttft": 0.74,
            "cost_pct": 1.5,
            "ttft_pct": 43.3,
            "score": 0.1974
        },
        {
            "name": "Phi-4",
            "quality": 66.6,
            "cost": 0.22,
            "ttft": 1.24,
            "cost_pct": 2.2,
            "ttft_pct": 72.2,
            "score": 0.2
        },
        {
            "name": "Llama 4 Scout",
            "quality": 54.4,
            "cost": 0.24,
            "ttft": 0.48,
            "cost_pct": 2.4,
            "ttft_pct": 28.0,
            "score": 0.2118
        },
        {
            "name": "Gemma 3 12B Instruct",
            "quality": 57.7,
            "cost": 0.0,
            "ttft": 0.44,
            "cost_pct": 0.0,
            "ttft_pct": 25.7,
            "score": 0.2169
        },
        {
            "name": "Phi-4 Mini Instruct",
            "quality": 53.1,
            "cost": 0.0,
            "ttft": 0.7,
            "cost_pct": 0.0,
            "ttft_pct": 41.0,
            "score": 0.2555
        }
    ],
    "qa_0_5": [
        {
            "name": "DeepSeek V3.2 Exp (Reasoning)",
            "quality": 91.4,
            "cost": 0.32,
            "ttft": 0.57,
            "cost_pct": 3.1,
            "ttft_pct": 33.3,
            "score": 0.1176
        },
        {
            "name": "Grok 3 mini Reasoning (high)",
            "quality": 88.0,
            "cost": 0.35,
            "ttft": 0.51,
            "cost_pct": 3.5,
            "ttft_pct": 29.6,
            "score": 0.1262
        },
        {
            "name": "Gemini 2.5 Flash-Lite Preview (Sep '25) (Reasoning)",
            "quality": 85.1,
            "cost": 0.17,
            "ttft": 0.51,
            "cost_pct": 1.7,
            "ttft_pct": 29.8,
            "score": 0.1338
        },
        {
            "name": "Grok 4.1 Fast (Reasoning)",
            "quality": 79.2,
            "cost": 0.28,
            "ttft": 0.63,
            "cost_pct": 2.8,
            "ttft_pct": 36.9,
            "score": 0.1486
        },
        {
            "name": "gpt-oss-120B (high)",
            "quality": 75.2,
            "cost": 0.26,
            "ttft": 0.32,
            "cost_pct": 2.6,
            "ttft_pct": 18.4,
            "score": 0.1588
        },
        {
            "name": "Qwen3 4B 2507 (Reasoning)",
            "quality": 72.0,
            "cost": 0.0,
            "ttft": 0.91,
            "cost_pct": 0.0,
            "ttft_pct": 52.8,
            "score": 0.167
        },
        {
            "name": "GLM-4.5-Air",
            "quality": 71.0,
            "cost": 0.42,
            "ttft": 0.69,
            "cost_pct": 4.2,
            "ttft_pct": 40.2,
            "score": 0.1694
        },
        {
            "name": "DeepSeek R1 0528 Qwen3 8B",
            "quality": 69.6,
            "cost": 0.07,
            "ttft": 0.77,
            "cost_pct": 0.7,
            "ttft_pct": 45.0,
            "score": 0.1732
        },
        {
            "name": "GPT-5 nano (high)",
            "quality": 69.5,
            "cost": 0.14,
            "ttft": 0.27,
            "cost_pct": 1.4,
            "ttft_pct": 15.6,
            "score": 0.1735
        },
        {
            "name": "Mistral Small 3.2",
            "quality": 63.7,
            "cost": 0.15,
            "ttft": 0.51,
            "cost_pct": 1.5,
            "ttft_pct": 29.8,
            "score": 0.1882
        }
    ],
    "qa_1_0": [
        {
            "name": "DeepSeek V3.1 Terminus (Reasoning)",
            "quality": 93.1,
            "cost": 0.8,
            "ttft": 0.44,
            "cost_pct": 8.0,
            "ttft_pct": 25.7,
            "score": 0.1133
        },
        {
            "name": "DeepSeek V3.2 Exp (Reasoning)",
            "quality": 91.4,
            "cost": 0.32,
            "ttft": 0.57,
            "cost_pct": 3.1,
            "ttft_pct": 33.3,
            "score": 0.1176
        },
        {
            "name": "Grok 3 mini Reasoning (high)",
            "quality": 88.0,
            "cost": 0.35,
            "ttft": 0.51,
            "cost_pct": 3.5,
            "ttft_pct": 29.6,
            "score": 0.1262
        },
        {
            "name": "Gemini 2.5 Flash-Lite Preview (Sep '25) (Reasoning)",
            "quality": 85.1,
            "cost": 0.17,
            "ttft": 0.51,
            "cost_pct": 1.7,
            "ttft_pct": 29.8,
            "score": 0.1338
        },
        {
            "name": "GPT-5 mini (high)",
            "quality": 84.3,
            "cost": 0.69,
            "ttft": 0.34,
            "cost_pct": 6.9,
            "ttft_pct": 19.9,
            "score": 0.1357
        },
        {
            "name": "Gemini 2.5 Flash Preview (Sep '25) (Reasoning)",
            "quality": 79.7,
            "cost": 0.85,
            "ttft": 0.63,
            "cost_pct": 8.5,
            "ttft_pct": 36.5,
            "score": 0.1474
        },
        {
            "name": "Grok 4.1 Fast (Reasoning)",
            "quality": 79.2,
            "cost": 0.28,
            "ttft": 0.63,
            "cost_pct": 2.8,
            "ttft_pct": 36.9,
            "score": 0.1486
        },
        {
            "name": "gpt-oss-120B (high)",
            "quality": 75.2,
            "cost": 0.26,
            "ttft": 0.32,
            "cost_pct": 2.6,
            "ttft_pct": 18.4,
            "score": 0.1588
        },
        {
            "name": "GLM-4.5-Air",
            "quality": 71.0,
            "cost": 0.42,
            "ttft": 0.69,
            "cost_pct": 4.2,
            "ttft_pct": 40.2,
            "score": 0.1694
        },
        {
            "name": "DeepSeek R1 0528 Qwen3 8B",
            "quality": 69.6,
            "cost": 0.07,
            "ttft": 0.77,
            "cost_pct": 0.7,
            "ttft_pct": 45.0,
            "score": 0.1732
        }
    ],
    "qa_2_0": [
        {
            "name": "DeepSeek V3.1 Terminus (Reasoning)",
            "quality": 93.1,
            "cost": 0.8,
            "ttft": 0.44,
            "cost_pct": 8.0,
            "ttft_pct": 25.7,
            "score": 0.1133
        },
        {
            "name": "DeepSeek V3.2 Exp (Reasoning)",
            "quality": 91.4,
            "cost": 0.32,
            "ttft": 0.57,
            "cost_pct": 3.1,
            "ttft_pct": 33.3,
            "score": 0.1176
        },
        {
            "name": "Grok 3 mini Reasoning (high)",
            "quality": 88.0,
            "cost": 0.35,
            "ttft": 0.51,
            "cost_pct": 3.5,
            "ttft_pct": 29.6,
            "score": 0.1262
        },
        {
            "name": "Gemini 2.5 Flash-Lite Preview (Sep '25) (Reasoning)",
            "quality": 85.1,
            "cost": 0.17,
            "ttft": 0.51,
            "cost_pct": 1.7,
            "ttft_pct": 29.8,
            "score": 0.1338
        },
        {
            "name": "GPT-5 mini (high)",
            "quality": 84.3,
            "cost": 0.69,
            "ttft": 0.34,
            "cost_pct": 6.9,
            "ttft_pct": 19.9,
            "score": 0.1357
        },
        {
            "name": "Gemini 2.5 Flash Preview (Sep '25) (Reasoning)",
            "quality": 79.7,
            "cost": 0.85,
            "ttft": 0.63,
            "cost_pct": 8.5,
            "ttft_pct": 36.5,
            "score": 0.1474
        },
        {
            "name": "Grok 4.1 Fast (Reasoning)",
            "quality": 79.2,
            "cost": 0.28,
            "ttft": 0.63,
            "cost_pct": 2.8,
            "ttft_pct": 36.9,
            "score": 0.1486
        },
        {
            "name": "gpt-oss-120B (high)",
            "quality": 75.2,
            "cost": 0.26,
            "ttft": 0.32,
            "cost_pct": 2.6,
            "ttft_pct": 18.4,
            "score": 0.1588
        },
        {
            "name": "GLM-4.5-Air",
            "quality": 71.0,
            "cost": 0.42,
            "ttft": 0.69,
            "cost_pct": 4.2,
            "ttft_pct": 40.2,
            "score": 0.1694
        },
        {
            "name": "Qwen3 4B 2507 (Reasoning)",
            "quality": 72.0,
            "cost": 0.0,
            "ttft": 0.91,
            "cost_pct": 0.0,
            "ttft_pct": 52.8,
            "score": 0.1728
        }
    ],
    "qa_5_0": [
        {
            "name": "DeepSeek V3.1 Terminus (Reasoning)",
            "quality": 93.1,
            "cost": 0.8,
            "ttft": 0.44,
            "cost_pct": 8.0,
            "ttft_pct": 25.7,
            "score": 0.1133
        },
        {
            "name": "GPT-5.1 (high)",
            "quality": 98.7,
            "cost": 3.44,
            "ttft": 0.33,
            "cost_pct": 34.4,
            "ttft_pct": 19.4,
            "score": 0.1172
        },
        {
            "name": "DeepSeek V3.2 Exp (Reasoning)",
            "quality": 91.4,
            "cost": 0.32,
            "ttft": 0.57,
            "cost_pct": 3.1,
            "ttft_pct": 33.3,
            "score": 0.1176
        },
        {
            "name": "Grok 3 mini Reasoning (high)",
            "quality": 88.0,
            "cost": 0.35,
            "ttft": 0.51,
            "cost_pct": 3.5,
            "ttft_pct": 29.6,
            "score": 0.1262
        },
        {
            "name": "Gemini 2.5 Pro",
            "quality": 93.6,
            "cost": 3.44,
            "ttft": 0.94,
            "cost_pct": 34.4,
            "ttft_pct": 55.0,
            "score": 0.1302
        },
        {
            "name": "Gemini 2.5 Flash-Lite Preview (Sep '25) (Reasoning)",
            "quality": 85.1,
            "cost": 0.17,
            "ttft": 0.51,
            "cost_pct": 1.7,
            "ttft_pct": 29.8,
            "score": 0.1338
        },
        {
            "name": "GPT-5 mini (high)",
            "quality": 84.3,
            "cost": 0.69,
            "ttft": 0.34,
            "cost_pct": 6.9,
            "ttft_pct": 19.9,
            "score": 0.1357
        },
        {
            "name": "Gemini 3 Pro Preview (high)",
            "quality": 99.5,
            "cost": 4.5,
            "ttft": 1.96,
            "cost_pct": 45.0,
            "ttft_pct": 114.5,
            "score": 0.1465
        },
        {
            "name": "Gemini 2.5 Flash Preview (Sep '25) (Reasoning)",
            "quality": 79.7,
            "cost": 0.85,
            "ttft": 0.63,
            "cost_pct": 8.5,
            "ttft_pct": 36.5,
            "score": 0.1474
        },
        {
            "name": "Grok 4.1 Fast (Reasoning)",
            "quality": 79.2,
            "cost": 0.28,
            "ttft": 0.63,
            "cost_pct": 2.8,
            "ttft_pct": 36.9,
            "score": 0.1486
        }
    ],
    "rag_0_1": [
        {
            "name": "Gemma 3 4B Instruct",
            "quality": 57.9,
            "cost": 0.0,
            "ttft": 0.29,
            "cost_pct": 0.0,
            "ttft_pct": 14.7,
            "score": 0.1819,
            "context_k": 1048
        },
        {
            "name": "Qwen3 4B 2507 (Reasoning)",
            "quality": 54.1,
            "cost": 0.0,
            "ttft": 0.91,
            "cost_pct": 0.0,
            "ttft_pct": 46.1,
            "score": 0.2,
            "context_k": 40
        },
        {
            "name": "DeepSeek R1 0528 Qwen3 8B",
            "quality": 49.0,
            "cost": 0.07,
            "ttft": 0.77,
            "cost_pct": 1.5,
            "ttft_pct": 39.3,
            "score": 0.2078,
            "context_k": 32
        },
        {
            "name": "Gemma 3 27B Instruct",
            "quality": 47.8,
            "cost": 0.0,
            "ttft": 0.41,
            "cost_pct": 0.0,
            "ttft_pct": 20.7,
            "score": 0.2114,
            "context_k": 131
        },
        {
            "name": "Gemma 3 12B Instruct",
            "quality": 44.9,
            "cost": 0.0,
            "ttft": 0.44,
            "cost_pct": 0.0,
            "ttft_pct": 22.4,
            "score": 0.2198,
            "context_k": 60
        },
        {
            "name": "Ministral 8B",
            "quality": 38.6,
            "cost": 0.1,
            "ttft": 0.72,
            "cost_pct": 2.2,
            "ttft_pct": 36.5,
            "score": 0.2381,
            "context_k": 131
        },
        {
            "name": "Ministral 3B",
            "quality": 37.6,
            "cost": 0.04,
            "ttft": 0.53,
            "cost_pct": 0.9,
            "ttft_pct": 26.8,
            "score": 0.2409,
            "context_k": 131
        },
        {
            "name": "Phi-4 Mini Instruct",
            "quality": 28.3,
            "cost": 0.0,
            "ttft": 0.7,
            "cost_pct": 0.0,
            "ttft_pct": 35.8,
            "score": 0.3,
            "context_k": 16
        }
    ],
    "rag_0_25": [
        {
            "name": "Gemini 2.5 Flash-Lite Preview (Sep '25) (Reasoning)",
            "quality": 82.8,
            "cost": 0.17,
            "ttft": 0.51,
            "cost_pct": 3.9,
            "ttft_pct": 26.0,
            "score": 0.1098,
            "context_k": 1048
        },
        {
            "name": "GPT-5 nano (high)",
            "quality": 68.0,
            "cost": 0.14,
            "ttft": 0.27,
            "cost_pct": 3.1,
            "ttft_pct": 13.6,
            "score": 0.1526,
            "context_k": 400
        },
        {
            "name": "Llama 4 Scout",
            "quality": 59.9,
            "cost": 0.24,
            "ttft": 0.48,
            "cost_pct": 5.4,
            "ttft_pct": 24.5,
            "score": 0.1763,
            "context_k": 327
        },
        {
            "name": "Gemma 3 4B Instruct",
            "quality": 57.9,
            "cost": 0.0,
            "ttft": 0.29,
            "cost_pct": 0.0,
            "ttft_pct": 14.7,
            "score": 0.1819,
            "context_k": 1048
        },
        {
            "name": "Qwen3 4B 2507 (Reasoning)",
            "quality": 54.1,
            "cost": 0.0,
            "ttft": 0.91,
            "cost_pct": 0.0,
            "ttft_pct": 46.1,
            "score": 0.1932,
            "context_k": 40
        },
        {
            "name": "DeepSeek R1 0528 Qwen3 8B",
            "quality": 49.0,
            "cost": 0.07,
            "ttft": 0.77,
            "cost_pct": 1.5,
            "ttft_pct": 39.3,
            "score": 0.2078,
            "context_k": 32
        },
        {
            "name": "Gemma 3 27B Instruct",
            "quality": 47.8,
            "cost": 0.0,
            "ttft": 0.41,
            "cost_pct": 0.0,
            "ttft_pct": 20.7,
            "score": 0.2114,
            "context_k": 131
        },
        {
            "name": "Mistral Small 3.2",
            "quality": 43.4,
            "cost": 0.15,
            "ttft": 0.51,
            "cost_pct": 3.3,
            "ttft_pct": 26.1,
            "score": 0.224,
            "context_k": 32
        },
        {
            "name": "Mistral Small 3.1",
            "quality": 41.0,
            "cost": 0.15,
            "ttft": 0.74,
            "cost_pct": 3.3,
            "ttft_pct": 37.8,
            "score": 0.2311,
            "context_k": 32
        },
        {
            "name": "Gemma 3 12B Instruct",
            "quality": 44.9,
            "cost": 0.0,
            "ttft": 0.44,
            "cost_pct": 0.0,
            "ttft_pct": 22.4,
            "score": 0.2321,
            "context_k": 60
        }
    ],
    "rag_0_5": [
        {
            "name": "Grok 4.1 Fast (Reasoning)",
            "quality": 87.6,
            "cost": 0.28,
            "ttft": 0.63,
            "cost_pct": 6.1,
            "ttft_pct": 32.2,
            "score": 0.0957,
            "context_k": 2000
        },
        {
            "name": "Gemini 2.5 Flash-Lite Preview (Sep '25) (Reasoning)",
            "quality": 82.8,
            "cost": 0.17,
            "ttft": 0.51,
            "cost_pct": 3.9,
            "ttft_pct": 26.0,
            "score": 0.1098,
            "context_k": 1048
        },
        {
            "name": "DeepSeek V3.2 Exp (Reasoning)",
            "quality": 74.1,
            "cost": 0.32,
            "ttft": 0.57,
            "cost_pct": 7.0,
            "ttft_pct": 29.1,
            "score": 0.1349,
            "context_k": 163
        },
        {
            "name": "Grok 3 mini Reasoning (high)",
            "quality": 68.6,
            "cost": 0.35,
            "ttft": 0.51,
            "cost_pct": 7.8,
            "ttft_pct": 25.8,
            "score": 0.151,
            "context_k": 131
        },
        {
            "name": "GPT-5 nano (high)",
            "quality": 68.0,
            "cost": 0.14,
            "ttft": 0.27,
            "cost_pct": 3.1,
            "ttft_pct": 13.6,
            "score": 0.1526,
            "context_k": 400
        },
        {
            "name": "Llama 4 Maverick",
            "quality": 66.3,
            "cost": 0.42,
            "ttft": 0.47,
            "cost_pct": 9.4,
            "ttft_pct": 23.7,
            "score": 0.1577,
            "context_k": 1048
        },
        {
            "name": "gpt-oss-120B (high)",
            "quality": 64.6,
            "cost": 0.26,
            "ttft": 0.32,
            "cost_pct": 5.8,
            "ttft_pct": 16.1,
            "score": 0.1625,
            "context_k": 131
        },
        {
            "name": "GLM-4.5-Air",
            "quality": 61.6,
            "cost": 0.42,
            "ttft": 0.69,
            "cost_pct": 9.4,
            "ttft_pct": 35.1,
            "score": 0.1728,
            "context_k": 131
        },
        {
            "name": "Llama 4 Scout",
            "quality": 59.9,
            "cost": 0.24,
            "ttft": 0.48,
            "cost_pct": 5.4,
            "ttft_pct": 24.5,
            "score": 0.1763,
            "context_k": 327
        },
        {
            "name": "Gemma 3 4B Instruct",
            "quality": 57.9,
            "cost": 0.0,
            "ttft": 0.29,
            "cost_pct": 0.0,
            "ttft_pct": 14.7,
            "score": 0.2013,
            "context_k": 1048
        }
    ],
    "rag_1_0": [
        {
            "name": "Grok 4.1 Fast (Reasoning)",
            "quality": 87.6,
            "cost": 0.28,
            "ttft": 0.63,
            "cost_pct": 6.1,
            "ttft_pct": 32.2,
            "score": 0.0957,
            "context_k": 2000
        },
        {
            "name": "Gemini 2.5 Flash-Lite Preview (Sep '25) (Reasoning)",
            "quality": 82.8,
            "cost": 0.17,
            "ttft": 0.51,
            "cost_pct": 3.9,
            "ttft_pct": 26.0,
            "score": 0.1098,
            "context_k": 1048
        },
        {
            "name": "GPT-5 mini (high)",
            "quality": 81.0,
            "cost": 0.69,
            "ttft": 0.34,
            "cost_pct": 15.3,
            "ttft_pct": 17.4,
            "score": 0.115,
            "context_k": 400
        },
        {
            "name": "DeepSeek V3.2 Exp (Reasoning)",
            "quality": 74.1,
            "cost": 0.32,
            "ttft": 0.57,
            "cost_pct": 7.0,
            "ttft_pct": 29.1,
            "score": 0.1349,
            "context_k": 163
        },
        {
            "name": "DeepSeek V3.1 Terminus (Reasoning)",
            "quality": 72.3,
            "cost": 0.8,
            "ttft": 0.44,
            "cost_pct": 17.8,
            "ttft_pct": 22.4,
            "score": 0.1404,
            "context_k": 131
        },
        {
            "name": "Grok 3 mini Reasoning (high)",
            "quality": 68.6,
            "cost": 0.35,
            "ttft": 0.51,
            "cost_pct": 7.8,
            "ttft_pct": 25.8,
            "score": 0.151,
            "context_k": 131
        },
        {
            "name": "GPT-5 nano (high)",
            "quality": 68.0,
            "cost": 0.14,
            "ttft": 0.27,
            "cost_pct": 3.1,
            "ttft_pct": 13.6,
            "score": 0.1526,
            "context_k": 400
        },
        {
            "name": "Llama 4 Maverick",
            "quality": 66.3,
            "cost": 0.42,
            "ttft": 0.47,
            "cost_pct": 9.4,
            "ttft_pct": 23.7,
            "score": 0.1577,
            "context_k": 1048
        },
        {
            "name": "gpt-oss-120B (high)",
            "quality": 64.6,
            "cost": 0.26,
            "ttft": 0.32,
            "cost_pct": 5.8,
            "ttft_pct": 16.1,
            "score": 0.1625,
            "context_k": 131
        },
        {
            "name": "GLM-4.5-Air",
            "quality": 61.6,
            "cost": 0.42,
            "ttft": 0.69,
            "cost_pct": 9.4,
            "ttft_pct": 35.1,
            "score": 0.1713,
            "context_k": 131
        }
    ],
    "rag_2_0": [
        {
            "name": "Grok 4.1 Fast (Reasoning)",
            "quality": 87.6,
            "cost": 0.28,
            "ttft": 0.63,
            "cost_pct": 6.1,
            "ttft_pct": 32.2,
            "score": 0.0957,
            "context_k": 2000
        },
        {
            "name": "Gemini 2.5 Flash-Lite Preview (Sep '25) (Reasoning)",
            "quality": 82.8,
            "cost": 0.17,
            "ttft": 0.51,
            "cost_pct": 3.9,
            "ttft_pct": 26.0,
            "score": 0.1098,
            "context_k": 1048
        },
        {
            "name": "GPT-5 mini (high)",
            "quality": 81.0,
            "cost": 0.69,
            "ttft": 0.34,
            "cost_pct": 15.3,
            "ttft_pct": 17.4,
            "score": 0.115,
            "context_k": 400
        },
        {
            "name": "DeepSeek V3.2 Exp (Reasoning)",
            "quality": 74.1,
            "cost": 0.32,
            "ttft": 0.57,
            "cost_pct": 7.0,
            "ttft_pct": 29.1,
            "score": 0.1349,
            "context_k": 163
        },
        {
            "name": "DeepSeek V3.1 Terminus (Reasoning)",
            "quality": 72.3,
            "cost": 0.8,
            "ttft": 0.44,
            "cost_pct": 17.8,
            "ttft_pct": 22.4,
            "score": 0.1404,
            "context_k": 131
        },
        {
            "name": "Grok 3 mini Reasoning (high)",
            "quality": 68.6,
            "cost": 0.35,
            "ttft": 0.51,
            "cost_pct": 7.8,
            "ttft_pct": 25.8,
            "score": 0.151,
            "context_k": 131
        },
        {
            "name": "GPT-5 nano (high)",
            "quality": 68.0,
            "cost": 0.14,
            "ttft": 0.27,
            "cost_pct": 3.1,
            "ttft_pct": 13.6,
            "score": 0.1526,
            "context_k": 400
        },
        {
            "name": "Llama 4 Maverick",
            "quality": 66.3,
            "cost": 0.42,
            "ttft": 0.47,
            "cost_pct": 9.4,
            "ttft_pct": 23.7,
            "score": 0.1577,
            "context_k": 1048
        },
        {
            "name": "gpt-oss-120B (high)",
            "quality": 64.6,
            "cost": 0.26,
            "ttft": 0.32,
            "cost_pct": 5.8,
            "ttft_pct": 16.1,
            "score": 0.1625,
            "context_k": 131
        },
        {
            "name": "GLM-4.5-Air",
            "quality": 61.6,
            "cost": 0.42,
            "ttft": 0.69,
            "cost_pct": 9.4,
            "ttft_pct": 35.1,
            "score": 0.1713,
            "context_k": 131
        }
    ],
    "rag_5_0": [
        {
            "name": "Grok 4.1 Fast (Reasoning)",
            "quality": 87.6,
            "cost": 0.28,
            "ttft": 0.63,
            "cost_pct": 6.1,
            "ttft_pct": 32.2,
            "score": 0.0957,
            "context_k": 2000
        },
        {
            "name": "Gemini 2.5 Flash-Lite Preview (Sep '25) (Reasoning)",
            "quality": 82.8,
            "cost": 0.17,
            "ttft": 0.51,
            "cost_pct": 3.9,
            "ttft_pct": 26.0,
            "score": 0.1098,
            "context_k": 1048
        },
        {
            "name": "GPT-5 mini (high)",
            "quality": 81.0,
            "cost": 0.69,
            "ttft": 0.34,
            "cost_pct": 15.3,
            "ttft_pct": 17.4,
            "score": 0.115,
            "context_k": 400
        },
        {
            "name": "DeepSeek V3.2 Exp (Reasoning)",
            "quality": 74.1,
            "cost": 0.32,
            "ttft": 0.57,
            "cost_pct": 7.0,
            "ttft_pct": 29.1,
            "score": 0.1349,
            "context_k": 163
        },
        {
            "name": "Gemini 2.5 Pro",
            "quality": 93.2,
            "cost": 3.44,
            "ttft": 0.94,
            "cost_pct": 76.4,
            "ttft_pct": 48.0,
            "score": 0.135,
            "context_k": 1048
        },
        {
            "name": "DeepSeek V3.1 Terminus (Reasoning)",
            "quality": 72.3,
            "cost": 0.8,
            "ttft": 0.44,
            "cost_pct": 17.8,
            "ttft_pct": 22.4,
            "score": 0.1404,
            "context_k": 131
        },
        {
            "name": "GLM-4.6 (Reasoning)",
            "quality": 70.0,
            "cost": 1,
            "ttft": 1.69,
            "cost_pct": 22.2,
            "ttft_pct": 85.9,
            "score": 0.1469,
            "context_k": 202
        },
        {
            "name": "GPT-5.1 (high)",
            "quality": 90.5,
            "cost": 3.44,
            "ttft": 0.33,
            "cost_pct": 76.4,
            "ttft_pct": 16.9,
            "score": 0.1476,
            "context_k": 400
        },
        {
            "name": "Grok 3 mini Reasoning (high)",
            "quality": 68.6,
            "cost": 0.35,
            "ttft": 0.51,
            "cost_pct": 7.8,
            "ttft_pct": 25.8,
            "score": 0.151,
            "context_k": 131
        },
        {
            "name": "GPT-5 nano (high)",
            "quality": 68.0,
            "cost": 0.14,
            "ttft": 0.27,
            "cost_pct": 3.1,
            "ttft_pct": 13.6,
            "score": 0.1526,
            "context_k": 400
        }
    ],
    "chatbot_0_1": [
        {
            "name": "DeepSeek R1 0528 Qwen3 8B",
            "quality": 65.7,
            "cost": 0.07,
            "ttft": 0.77,
            "cost_pct": 9.9,
            "ttft_pct": 225.7,
            "score": 0.19
        },
        {
            "name": "Qwen3 4B 2507 (Reasoning)",
            "quality": 72.4,
            "cost": 0.0,
            "ttft": 0.91,
            "cost_pct": 0.0,
            "ttft_pct": 265.2,
            "score": 0.2
        },
        {
            "name": "Gemma 3 12B Instruct",
            "quality": 48.2,
            "cost": 0.0,
            "ttft": 0.44,
            "cost_pct": 0.0,
            "ttft_pct": 128.9,
            "score": 0.2327
        },
        {
            "name": "Gemma 3 27B Instruct",
            "quality": 44.2,
            "cost": 0.0,
            "ttft": 0.41,
            "cost_pct": 0.0,
            "ttft_pct": 118.8,
            "score": 0.2424
        },
        {
            "name": "Phi-4 Mini Instruct",
            "quality": 39.0,
            "cost": 0.0,
            "ttft": 0.7,
            "cost_pct": 0.0,
            "ttft_pct": 205.8,
            "score": 0.2609
        },
        {
            "name": "Ministral 8B",
            "quality": 25.1,
            "cost": 0.1,
            "ttft": 0.72,
            "cost_pct": 14.5,
            "ttft_pct": 209.5,
            "score": 0.2888
        },
        {
            "name": "Ministral 3B",
            "quality": 24.2,
            "cost": 0.04,
            "ttft": 0.53,
            "cost_pct": 5.8,
            "ttft_pct": 154.1,
            "score": 0.2912
        },
        {
            "name": "Gemma 3 4B Instruct",
            "quality": 33.0,
            "cost": 0.0,
            "ttft": 0.29,
            "cost_pct": 0.0,
            "ttft_pct": 84.2,
            "score": 0.3
        }
    ],
    "chatbot_0_25": [
        {
            "name": "Gemini 2.5 Flash-Lite Preview (Sep '25) (Reasoning)",
            "quality": 80.2,
            "cost": 0.17,
            "ttft": 0.51,
            "cost_pct": 25.4,
            "ttft_pct": 149.6,
            "score": 0.1546
        },
        {
            "name": "GPT-5 nano (high)",
            "quality": 74.4,
            "cost": 0.14,
            "ttft": 0.27,
            "cost_pct": 20.1,
            "ttft_pct": 78.2,
            "score": 0.1689
        },
        {
            "name": "Qwen3 4B 2507 (Reasoning)",
            "quality": 72.4,
            "cost": 0.0,
            "ttft": 0.91,
            "cost_pct": 0.0,
            "ttft_pct": 265.2,
            "score": 0.1737
        },
        {
            "name": "DeepSeek R1 0528 Qwen3 8B",
            "quality": 65.7,
            "cost": 0.07,
            "ttft": 0.77,
            "cost_pct": 9.9,
            "ttft_pct": 225.7,
            "score": 0.19
        },
        {
            "name": "Mistral Small 3.2",
            "quality": 55.9,
            "cost": 0.15,
            "ttft": 0.51,
            "cost_pct": 21.8,
            "ttft_pct": 149.8,
            "score": 0.2138
        },
        {
            "name": "Phi-4",
            "quality": 54.5,
            "cost": 0.22,
            "ttft": 1.24,
            "cost_pct": 31.8,
            "ttft_pct": 362.3,
            "score": 0.2174
        },
        {
            "name": "Mistral Small 3.1",
            "quality": 52.4,
            "cost": 0.15,
            "ttft": 0.74,
            "cost_pct": 21.8,
            "ttft_pct": 217.5,
            "score": 0.2224
        },
        {
            "name": "Llama 4 Scout",
            "quality": 50.6,
            "cost": 0.24,
            "ttft": 0.48,
            "cost_pct": 35.0,
            "ttft_pct": 140.7,
            "score": 0.2268
        },
        {
            "name": "Gemma 3 12B Instruct",
            "quality": 48.2,
            "cost": 0.0,
            "ttft": 0.44,
            "cost_pct": 0.0,
            "ttft_pct": 128.9,
            "score": 0.2367
        },
        {
            "name": "Gemma 3 27B Instruct",
            "quality": 44.2,
            "cost": 0.0,
            "ttft": 0.41,
            "cost_pct": 0.0,
            "ttft_pct": 118.8,
            "score": 0.2606
        }
    ],
    "chatbot_0_5": [
        {
            "name": "Grok 4.1 Fast (Reasoning)",
            "quality": 96.6,
            "cost": 0.28,
            "ttft": 0.63,
            "cost_pct": 40.0,
            "ttft_pct": 185.0,
            "score": 0.1146
        },
        {
            "name": "DeepSeek V3.2 Exp (Reasoning)",
            "quality": 96.3,
            "cost": 0.32,
            "ttft": 0.57,
            "cost_pct": 45.8,
            "ttft_pct": 167.2,
            "score": 0.1154
        },
        {
            "name": "Grok 3 mini Reasoning (high)",
            "quality": 90.5,
            "cost": 0.35,
            "ttft": 0.51,
            "cost_pct": 50.9,
            "ttft_pct": 148.5,
            "score": 0.1295
        },
        {
            "name": "gpt-oss-120B (high)",
            "quality": 88.0,
            "cost": 0.26,
            "ttft": 0.32,
            "cost_pct": 38.2,
            "ttft_pct": 92.2,
            "score": 0.1357
        },
        {
            "name": "Gemini 2.5 Flash-Lite Preview (Sep '25) (Reasoning)",
            "quality": 80.2,
            "cost": 0.17,
            "ttft": 0.51,
            "cost_pct": 25.4,
            "ttft_pct": 149.6,
            "score": 0.1546
        },
        {
            "name": "GLM-4.5-Air",
            "quality": 75.0,
            "cost": 0.42,
            "ttft": 0.69,
            "cost_pct": 61.8,
            "ttft_pct": 201.9,
            "score": 0.1673
        },
        {
            "name": "GPT-5 nano (high)",
            "quality": 74.4,
            "cost": 0.14,
            "ttft": 0.27,
            "cost_pct": 20.1,
            "ttft_pct": 78.2,
            "score": 0.1689
        },
        {
            "name": "Qwen3 4B 2507 (Reasoning)",
            "quality": 72.4,
            "cost": 0.0,
            "ttft": 0.91,
            "cost_pct": 0.0,
            "ttft_pct": 265.2,
            "score": 0.1759
        },
        {
            "name": "DeepSeek R1 0528 Qwen3 8B",
            "quality": 65.7,
            "cost": 0.07,
            "ttft": 0.77,
            "cost_pct": 9.9,
            "ttft_pct": 225.7,
            "score": 0.19
        },
        {
            "name": "Llama 4 Maverick",
            "quality": 61.2,
            "cost": 0.42,
            "ttft": 0.47,
            "cost_pct": 61.3,
            "ttft_pct": 136.2,
            "score": 0.201
        }
    ],
    "chatbot_1_0": [
        {
            "name": "Grok 4.1 Fast (Reasoning)",
            "quality": 96.6,
            "cost": 0.28,
            "ttft": 0.63,
            "cost_pct": 40.0,
            "ttft_pct": 185.0,
            "score": 0.1146
        },
        {
            "name": "DeepSeek V3.2 Exp (Reasoning)",
            "quality": 96.3,
            "cost": 0.32,
            "ttft": 0.57,
            "cost_pct": 45.8,
            "ttft_pct": 167.2,
            "score": 0.1154
        },
        {
            "name": "GPT-5 mini (high)",
            "quality": 100.0,
            "cost": 0.69,
            "ttft": 0.34,
            "cost_pct": 100.0,
            "ttft_pct": 100.0,
            "score": 0.125
        },
        {
            "name": "Grok 3 mini Reasoning (high)",
            "quality": 90.5,
            "cost": 0.35,
            "ttft": 0.51,
            "cost_pct": 50.9,
            "ttft_pct": 148.5,
            "score": 0.1295
        },
        {
            "name": "DeepSeek V3.1 Terminus (Reasoning)",
            "quality": 99.1,
            "cost": 0.8,
            "ttft": 0.44,
            "cost_pct": 116.3,
            "ttft_pct": 128.9,
            "score": 0.1345
        },
        {
            "name": "gpt-oss-120B (high)",
            "quality": 88.0,
            "cost": 0.26,
            "ttft": 0.32,
            "cost_pct": 38.2,
            "ttft_pct": 92.2,
            "score": 0.1357
        },
        {
            "name": "Gemini 2.5 Flash Preview (Sep '25) (Reasoning)",
            "quality": 86.5,
            "cost": 0.85,
            "ttft": 0.63,
            "cost_pct": 123.5,
            "ttft_pct": 183.4,
            "score": 0.1393
        },
        {
            "name": "Gemini 2.5 Flash-Lite Preview (Sep '25) (Reasoning)",
            "quality": 80.2,
            "cost": 0.17,
            "ttft": 0.51,
            "cost_pct": 25.4,
            "ttft_pct": 149.6,
            "score": 0.1546
        },
        {
            "name": "GLM-4.5-Air",
            "quality": 75.0,
            "cost": 0.42,
            "ttft": 0.69,
            "cost_pct": 61.8,
            "ttft_pct": 201.9,
            "score": 0.1673
        },
        {
            "name": "GPT-5 nano (high)",
            "quality": 74.4,
            "cost": 0.14,
            "ttft": 0.27,
            "cost_pct": 20.1,
            "ttft_pct": 78.2,
            "score": 0.1689
        }
    ],
    "chatbot_2_0": [
        {
            "name": "Grok 4.1 Fast (Reasoning)",
            "quality": 96.6,
            "cost": 0.28,
            "ttft": 0.63,
            "cost_pct": 40.0,
            "ttft_pct": 185.0,
            "score": 0.1146
        },
        {
            "name": "DeepSeek V3.2 Exp (Reasoning)",
            "quality": 96.3,
            "cost": 0.32,
            "ttft": 0.57,
            "cost_pct": 45.8,
            "ttft_pct": 167.2,
            "score": 0.1154
        },
        {
            "name": "GPT-5 mini (high)",
            "quality": 100.0,
            "cost": 0.69,
            "ttft": 0.34,
            "cost_pct": 100.0,
            "ttft_pct": 100.0,
            "score": 0.125
        },
        {
            "name": "Grok 3 mini Reasoning (high)",
            "quality": 90.5,
            "cost": 0.35,
            "ttft": 0.51,
            "cost_pct": 50.9,
            "ttft_pct": 148.5,
            "score": 0.1295
        },
        {
            "name": "DeepSeek V3.1 Terminus (Reasoning)",
            "quality": 99.1,
            "cost": 0.8,
            "ttft": 0.44,
            "cost_pct": 116.3,
            "ttft_pct": 128.9,
            "score": 0.1345
        },
        {
            "name": "gpt-oss-120B (high)",
            "quality": 88.0,
            "cost": 0.26,
            "ttft": 0.32,
            "cost_pct": 38.2,
            "ttft_pct": 92.2,
            "score": 0.1357
        },
        {
            "name": "Gemini 2.5 Flash Preview (Sep '25) (Reasoning)",
            "quality": 86.5,
            "cost": 0.85,
            "ttft": 0.63,
            "cost_pct": 123.5,
            "ttft_pct": 183.4,
            "score": 0.1393
        },
        {
            "name": "Gemini 2.5 Flash-Lite Preview (Sep '25) (Reasoning)",
            "quality": 80.2,
            "cost": 0.17,
            "ttft": 0.51,
            "cost_pct": 25.4,
            "ttft_pct": 149.6,
            "score": 0.1546
        },
        {
            "name": "GLM-4.5-Air",
            "quality": 75.0,
            "cost": 0.42,
            "ttft": 0.69,
            "cost_pct": 61.8,
            "ttft_pct": 201.9,
            "score": 0.1673
        },
        {
            "name": "GPT-5 nano (high)",
            "quality": 74.4,
            "cost": 0.14,
            "ttft": 0.27,
            "cost_pct": 20.1,
            "ttft_pct": 78.2,
            "score": 0.1689
        }
    ],
    "chatbot_5_0": [
        {
            "name": "Grok 4.1 Fast (Reasoning)",
            "quality": 96.6,
            "cost": 0.28,
            "ttft": 0.63,
            "cost_pct": 40.0,
            "ttft_pct": 185.0,
            "score": 0.1146
        },
        {
            "name": "DeepSeek V3.2 Exp (Reasoning)",
            "quality": 96.3,
            "cost": 0.32,
            "ttft": 0.57,
            "cost_pct": 45.8,
            "ttft_pct": 167.2,
            "score": 0.1154
        },
        {
            "name": "GPT-5 mini (high)",
            "quality": 100.0,
            "cost": 0.69,
            "ttft": 0.34,
            "cost_pct": 100.0,
            "ttft_pct": 100.0,
            "score": 0.125
        },
        {
            "name": "Grok 3 mini Reasoning (high)",
            "quality": 90.5,
            "cost": 0.35,
            "ttft": 0.51,
            "cost_pct": 50.9,
            "ttft_pct": 148.5,
            "score": 0.1295
        },
        {
            "name": "DeepSeek V3.1 Terminus (Reasoning)",
            "quality": 99.1,
            "cost": 0.8,
            "ttft": 0.44,
            "cost_pct": 116.3,
            "ttft_pct": 128.9,
            "score": 0.1345
        },
        {
            "name": "gpt-oss-120B (high)",
            "quality": 88.0,
            "cost": 0.26,
            "ttft": 0.32,
            "cost_pct": 38.2,
            "ttft_pct": 92.2,
            "score": 0.1357
        },
        {
            "name": "Gemini 2.5 Flash Preview (Sep '25) (Reasoning)",
            "quality": 86.5,
            "cost": 0.85,
            "ttft": 0.63,
            "cost_pct": 123.5,
            "ttft_pct": 183.4,
            "score": 0.1393
        },
        {
            "name": "GLM-4.6 (Reasoning)",
            "quality": 85.4,
            "cost": 1,
            "ttft": 1.69,
            "cost_pct": 145.3,
            "ttft_pct": 493.9,
            "score": 0.1481
        },
        {
            "name": "Gemini 2.5 Flash-Lite Preview (Sep '25) (Reasoning)",
            "quality": 80.2,
            "cost": 0.17,
            "ttft": 0.51,
            "cost_pct": 25.4,
            "ttft_pct": 149.6,
            "score": 0.1546
        },
        {
            "name": "GLM-4.5-Air",
            "quality": 75.0,
            "cost": 0.42,
            "ttft": 0.69,
            "cost_pct": 61.8,
            "ttft_pct": 201.9,
            "score": 0.1673
        }
    ]
};

// Budget key mapping
const BUDGET_KEYS = {0.10: '0_1', 0.25: '0_25', 0.50: '0_5', 1.00: '1_0', 2.00: '2_0', 5.00: '5_0'};

class BudgetDemo {
    constructor() {
        this.budgetSlider = document.getElementById('budget-slider');
        this.budgetDisplay = document.getElementById('budget-display');
        this.resultsList = document.getElementById('budget-results-list');
        this.resultsCount = document.getElementById('budget-results-count');
        this.opensourceCheckbox = document.getElementById('budget-opensource-only');
        this.baselineModel = document.getElementById('budget-baseline-model');
        
        if (!this.budgetSlider) return;
        
        this.useCase = 'coding';
        this.opensourceOnly = false;
        
        this.setupEventListeners();
        this.updateBaselineDisplay();
        this.updateResults();
    }
    
    updateBaselineDisplay() {
        if (this.baselineModel) {
            this.baselineModel.textContent = USE_CASE_BASELINES[this.useCase] || 'GPT-5.1 (high)';
        }
    }
    
    setupEventListeners() {
        // Budget slider
        this.budgetSlider.addEventListener('input', () => {
            const idx = parseInt(this.budgetSlider.value);
            const budget = BUDGET_VALUES[idx];
            this.budgetDisplay.textContent = `$${budget.toFixed(2)}/M tokens`;
            this.updateResults();
        });
        
        // Open source checkbox
        if (this.opensourceCheckbox) {
            this.opensourceCheckbox.addEventListener('change', () => {
                this.opensourceOnly = this.opensourceCheckbox.checked;
                this.updateResults();
            });
        }
        
        // Use case buttons (budget-specific)
        document.querySelectorAll('.budget-use-case .use-case-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.budget-use-case .use-case-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                this.useCase = btn.dataset.usecase;
                this.updateBaselineDisplay();
                this.updateResults();
            });
        });
    }
    
    updateResults() {
        const budgetIdx = parseInt(this.budgetSlider.value);
        const maxBudget = BUDGET_VALUES[budgetIdx];
        
        // Get the key for precomputed rankings
        const budgetKey = BUDGET_KEYS[maxBudget];
        const rankingKey = `${this.useCase}_${budgetKey}`;
        
        // Fetch precomputed HYBRID rankings for this budget + use case
        let rankedModels = BUDGET_RANKINGS[rankingKey] || [];
        
        // Filter by open source if checkbox is checked
        if (this.opensourceOnly) {
            rankedModels = rankedModels.filter(model => isOpenSource(model.name));
        }
        
        if (rankedModels.length === 0) {
            this.resultsCount.textContent = 'No models in budget';
            this.resultsList.innerHTML = `
                <div class="no-results">
                    <div class="no-results-icon">💸</div>
                    <div class="no-results-text">
                        <p>No models available at $${maxBudget.toFixed(2)}/M or less.</p>
                        <p class="no-results-hint">Try increasing your budget.</p>
                    </div>
                </div>
            `;
            return;
        }
        
        // Calculate savings vs baseline
        const baselineCost = BASELINE.cost;
        const baselineTTFT = BASELINE.ttft;
        
        this.resultsCount.textContent = `${rankedModels.length} models within budget (HYBRID ranked)`;
        
        this.resultsList.innerHTML = rankedModels.slice(0, 8).map((model, index) => {
            const costSavings = ((baselineCost - model.cost) / baselineCost * 100).toFixed(0);
            const latencySavings = ((baselineTTFT - model.ttft) / baselineTTFT * 100).toFixed(0);
            const isWinner = index === 0;
            const isOSS = isOpenSource(model.name);
            const ossBadge = isOSS ? '<span class="oss-badge">OSS</span>' : '';
            
            return `
                <div class="result-item ${isWinner ? 'winner budget-winner' : ''}">
                    <div class="result-rank ${isWinner ? 'meets-all' : ''}">${isWinner ? '🏆' : index + 1}</div>
                    <div class="result-info">
                        <div class="result-name-row">
                            <span class="result-model-name">${model.name}</span>
                            ${ossBadge}
                            ${isWinner ? '<span class="best-value-badge">Best for Budget</span>' : ''}
                        </div>
                        <span class="result-metrics">
                            <span class="metric pass">Q:${model.quality.toFixed(0)}%</span> | 
                            <span class="metric pass">$${model.cost.toFixed(2)}/M ↓${costSavings}%</span> | 
                            ${model.ttft.toFixed(2)}s ${latencySavings > 0 ? `↓${latencySavings}% faster` : (latencySavings < 0 ? `↑${Math.abs(latencySavings)}% slower` : '')}
                            ${model.context_k ? ` | <span class="metric context">📄 ${model.context_k}K</span>` : ''}
                        </span>
                    </div>
                    <div class="result-savings">
                        <span class="hybrid-score-badge">Score: ${model.score.toFixed(4)}</span>
                    </div>
                </div>
            `;
        }).join('');
    }
}

// Demo Mode Tab Switching
function initDemoModeTabs() {
    const tabs = document.querySelectorAll('.demo-mode-tab');
    const constraintsMode = document.getElementById('constraints-mode');
    const budgetMode = document.getElementById('budget-mode');
    
    if (!tabs.length || !constraintsMode || !budgetMode) return;
    
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            tabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            
            const mode = tab.dataset.mode;
            if (mode === 'constraints') {
                constraintsMode.classList.remove('hidden');
                budgetMode.classList.add('hidden');
            } else {
                constraintsMode.classList.add('hidden');
                budgetMode.classList.remove('hidden');
            }
        });
    });
}

// ============================================
// Query Demo Tabs
// ============================================
function initQueryTabs() {
    const tabs = document.querySelectorAll('.query-tab');
    const contents = document.querySelectorAll('.query-tab-content');
    
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            // Update active tab
            tabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            
            // Update active content
            const targetId = 'tab-' + tab.dataset.tab;
            contents.forEach(content => {
                content.classList.toggle('active', content.id === targetId);
            });
        });
    });
}

// ============================================
// Prompt Classification Animation
// ============================================
class PromptClassificationAnimation {
    constructor() {
        // Real HYBRID optimization recommendations from the algorithm
        this.prompts = [
            {
                text: "Write a Python function to calculate fibonacci with memoization",
                category: "💻 Coding",
                complexity: "📊 Medium",
                language: "🐍 Python",
                model: "GPT-5 mini (high)",
                stats: { quality: 88, cost: 0.25, latency: 0.3 },
                reason: "HYBRID #1 for Coding • Strong benchmarks • 87% cost savings"
            },
            {
                text: "Build a customer churn prediction model with scikit-learn",
                category: "📊 Data Science",
                complexity: "📊 High",
                language: "🐍 Python",
                model: "GPT-5 mini (high)",
                stats: { quality: 88, cost: 0.25, latency: 0.3 },
                reason: "HYBRID #1 for Data Science • Excellent math scores • Low cost"
            },
            {
                text: "Write a story about a robot discovering emotions",
                category: "✨ Creative",
                complexity: "📊 Medium",
                language: "📝 Prose",
                model: "Grok 4.1 Fast (Reasoning)",
                stats: { quality: 86, cost: 0.20, latency: 15.9 },
                reason: "HYBRID #1 for Creative • High net dominance • 90% cost savings"
            },
            {
                text: "Explain quantum entanglement for a high school student",
                category: "🎯 General",
                complexity: "📊 Medium",
                language: "📝 Explanation",
                model: "GPT-5 (high)",
                stats: { quality: 92, cost: 1.25, latency: 0.3 },
                reason: "HYBRID #1 for General • Highest intelligence index • Fast"
            },
        ];
        
        this.currentIndex = 0;
        this.charIndex = 0;
        this.isTyping = false;
        this.isDeleting = false;
        this.isPaused = false;
        
        this.promptElement = document.getElementById('user-prompt-text');
        this.resultElement = document.getElementById('prompt-result');
        this.taskTagElement = document.getElementById('prompt-task-tag');
        this.modelNameElement = document.getElementById('prompt-model-name');
        this.modelStatsElement = document.getElementById('prompt-model-stats');
        this.legendElement = document.getElementById('prompt-stats-legend');
        
        // Savings elements (shared with Natural Language Constraints tab)
        this.savingsElement = document.getElementById('hero-savings');
        this.savingsCost = document.getElementById('savings-cost');
        this.savingsCostDetail = document.getElementById('savings-cost-detail');
        this.savingsLatency = document.getElementById('savings-latency');
        this.savingsLatencyDetail = document.getElementById('savings-latency-detail');
        this.savingsValue = document.getElementById('savings-value');
        
        if (!this.promptElement) return;
        
        this.startAnimation();
    }
    
    startAnimation() {
        this.typePrompt();
    }
    
    typePrompt() {
        const prompt = this.prompts[this.currentIndex];
        const fullText = prompt.text;
        
        if (!this.isTyping && !this.isPaused && !this.isDeleting) {
            // Start typing
            this.isTyping = true;
            this.charIndex = 0;
            this.promptElement.textContent = '';
        }
        
        if (this.isTyping) {
            // Type next character
            this.promptElement.textContent = fullText.substring(0, this.charIndex + 1);
            this.charIndex++;
            
            if (this.charIndex >= fullText.length) {
                // Finished typing, show results
                this.isTyping = false;
                this.showResults(prompt);
                this.isPaused = true;
                
                // Wait 5 seconds (same as Natural Language Constraints) then start deleting
                setTimeout(() => {
                    this.isPaused = false;
                    this.isDeleting = true;
                    this.deletePrompt();
                }, 5000);
                return;
            }
            
            // Match typing speed of Natural Language Constraints (60-100ms per char)
            setTimeout(() => this.typePrompt(), 60 + Math.random() * 40);
        }
    }
    
    deletePrompt() {
        const prompt = this.prompts[this.currentIndex];
        const fullText = prompt.text;
        
        if (this.isDeleting) {
            // Delete character
            this.promptElement.textContent = fullText.substring(0, this.charIndex);
            this.charIndex--;
            
            if (this.charIndex <= 0) {
                // Finished deleting, hide results and move to next
                this.isDeleting = false;
                this.hideResults();
                this.currentIndex = (this.currentIndex + 1) % this.prompts.length;
                
                // Brief pause then start typing next prompt
                setTimeout(() => this.typePrompt(), 500);
                return;
            }
            
            // Delete speed (faster than typing)
            setTimeout(() => this.deletePrompt(), 25);
        }
    }
    
    hideResults() {
        if (this.resultElement) {
            this.resultElement.classList.remove('visible');
        }
        if (this.legendElement) {
            this.legendElement.classList.remove('visible');
        }
        if (this.savingsElement) {
            this.savingsElement.classList.remove('visible');
        }
    }
    
    showResults(prompt) {
        // Get task class for styling
        const taskClass = prompt.category.includes('Coding') ? 'coding' :
                          prompt.category.includes('Data') ? 'data-science' :
                          prompt.category.includes('Creative') ? 'creative' : 'general';
        
        // Update task tag
        if (this.taskTagElement) {
            this.taskTagElement.textContent = prompt.category;
            this.taskTagElement.className = 'result-task-tag ' + taskClass;
        }
        
        // Update model name
        if (this.modelNameElement) {
            this.modelNameElement.textContent = prompt.model;
        }
        
        // Calculate savings
        const baseline = BASELINE;
        const costSavings = ((baseline.cost - prompt.stats.cost) / baseline.cost) * 100;
        const costSaved = baseline.cost - prompt.stats.cost;
        const latencyImprovement = ((baseline.ttft - prompt.stats.latency) / baseline.ttft) * 100;
        const baselineValue = 100 / baseline.cost;
        const recommendedValue = prompt.stats.quality / prompt.stats.cost;
        const valueMultiplier = recommendedValue / baselineValue;
        
        // Build stats string with inline savings
        let statsText = `Q:${prompt.stats.quality}% | $${prompt.stats.cost.toFixed(2)}/M`;
        if (costSavings > 0) statsText += ` ↓${costSavings.toFixed(0)}%`;
        statsText += ` | ${prompt.stats.latency}s`;
        if (latencyImprovement > 0) statsText += ` ↓${latencyImprovement.toFixed(0)}%`;
        
        if (this.modelStatsElement) {
            this.modelStatsElement.textContent = statsText;
        }
        
        // Update savings boxes (same as Natural Language Constraints)
        if (this.savingsCost) {
            this.savingsCost.textContent = costSavings.toFixed(1) + '%';
        }
        if (this.savingsCostDetail) {
            this.savingsCostDetail.textContent = `($${costSaved.toFixed(2)} saved per 1M tokens)`;
        }
        if (this.savingsLatency) {
            const sign = latencyImprovement > 0 ? '-' : '+';
            this.savingsLatency.textContent = sign + Math.abs(latencyImprovement).toFixed(0) + '%';
        }
        if (this.savingsLatencyDetail) {
            this.savingsLatencyDetail.textContent = `(${prompt.stats.latency.toFixed(1)}s vs ${baseline.ttft.toFixed(1)}s TTFT)`;
        }
        if (this.savingsValue) {
            this.savingsValue.textContent = valueMultiplier.toFixed(1) + 'x';
        }
        
        // Show results, legend, and savings together
        if (this.resultElement) {
            this.resultElement.classList.add('visible');
        }
        if (this.legendElement) {
            this.legendElement.classList.add('visible');
        }
        if (this.savingsElement) {
            this.savingsElement.classList.add('visible');
        }
    }
}

// ============================================
// Code Tabs
// ============================================
function initCodeTabs() {
    document.querySelectorAll('.code-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            document.querySelectorAll('.code-tab').forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            
            document.querySelectorAll('.code-block').forEach(block => block.classList.add('hidden'));
            document.getElementById('code-' + tab.dataset.tab).classList.remove('hidden');
        });
    });
}

// ============================================
// Stats Counter Animation
// ============================================
function animateStats() {
    const stats = document.querySelectorAll('.stat-number');
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const target = parseInt(entry.target.dataset.target);
                animateNumber(entry.target, target);
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.5 });
    
    stats.forEach(stat => observer.observe(stat));
}

function animateNumber(element, target) {
    let current = 0;
    const increment = target / 60;
    const timer = setInterval(() => {
        current += increment;
        if (current >= target) {
            element.textContent = target;
            clearInterval(timer);
        } else {
            element.textContent = Math.floor(current);
        }
    }, 16);
}

// ============================================
// Copy Install Command
// ============================================
function copyInstall() {
    navigator.clipboard.writeText('pip install banditgpt');
    const btn = document.querySelector('.copy-btn');
    btn.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 6L9 17l-5-5"/></svg>';
    setTimeout(() => {
        btn.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>';
    }, 2000);
}

// ============================================
// Smooth Scroll
// ============================================
function initSmoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        });
    });
}

// ============================================
// Initialize Everything
// ============================================
document.addEventListener('DOMContentLoaded', () => {
    // Initialize animations
    new ParticleAnimation('particle-canvas');
    new QueryTypingAnimation();
    new PromptClassificationAnimation();
    new InteractiveDemo();
    new BudgetDemo();
    
    // Initialize UI components
    initQueryTabs();
    initCodeTabs();
    initDemoModeTabs();
    animateStats();
    initSmoothScroll();
    
    console.log('🚀 BanditGPT landing page initialized');
});

