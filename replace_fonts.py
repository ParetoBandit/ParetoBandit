import re

with open('experiments/02_figure/generate_figure2_architecture.py', 'r') as f:
    content = f.read()

# Helper function defaults
content = content.replace('lw=1.4, fs=9, fw="bold", z=3):', 'lw=1.4, fs=11, fw="bold", z=3):')
content = content.replace('tc="#1a1a1a", tc2=None, lw=1.4, fs=9, fs2=7, fw="bold", z=3):', 'tc="#1a1a1a", tc2=None, lw=1.4, fs=11, fs2=9, fw="bold", z=3):')
content = content.replace('def txt(x, y, s, fs=7, c=None, fw="normal", ha="center", va="center",', 'def txt(x, y, s, fs=9, c=None, fw="normal", ha="center", va="center",')

# Specific text replacements, replacing exactly and avoiding double matching
# We will use re.sub with word boundaries or exact strings. 

replacements = {
    r'fs=6,\s*c=PAL\["lt"\]': r'fs=8, c=PAL["lt"]',
    r'fs=6.5,\s*fw="bold",\s*c=PAL\["sky"\]': r'fs=8.5, fw="bold", c=PAL["sky"]',
    r'fs=6,\s*c=PAL\["md"\]': r'fs=8, c=PAL["md"]',
    r'fs=5.5,\s*c=PAL\["md"\]': r'fs=7.5, c=PAL["md"]',
    r'fs=8.5\)': r'fs=10.5)',
    r'fs=9,\s*c=PAL\["md"\]': r'fs=11, c=PAL["md"]',
    r'fs=8,\s*c=PAL\["lt"\]': r'fs=10, c=PAL["lt"]',
    r'fs=8,\s*c=PAL\["vermillion"\]': r'fs=10, c=PAL["vermillion"]',
    r'fs=8.5,\s*fw="bold",\s*c=PAL\["teal"\]': r'fs=10.5, fw="bold", c=PAL["teal"]',
    r'fs=6.5,\s*c=PAL\["md"\]': r'fs=8.5, c=PAL["md"]',
    r'fs=8.5,\s*fw="bold",\s*c="#b07d00"': r'fs=10.5, fw="bold", c="#b07d00"',
    r'fs=8,\s*c=PAL\["teal"\]': r'fs=10, c=PAL["teal"]',
    r'fs=8,\s*c="#b07d00"': r'fs=10, c="#b07d00"',
    r'fs=8.5,\s*fs2=6': r'fs=10.5, fs2=8',
    r'fs=5.5,\s*c=PAL\["lt"\]': r'fs=7.5, c=PAL["lt"]',
    r'fs=6,\s*fw="bold"': r'fs=8, fw="bold"',
    r'fs=5,\s*c="#aaa"': r'fs=7, c="#aaa"',
    r'fs=6.5,\s*c=PAL\["rpur"\]': r'fs=8.5, c=PAL["rpur"]',
    r'fs=7,\s*c=PAL\["rpur"\]': r'fs=9, c=PAL["rpur"]',
    r'fs=6,\s*fw="bold",\s*c=PAL\["dk"\]': r'fs=8, fw="bold", c=PAL["dk"]',
    r'fs=7,\s*fw="bold",\s*c=PAL\["dk"\]': r'fs=9, fw="bold", c=PAL["dk"]',
    r'fs=6,\s*c=PAL\["dk"\]': r'fs=8, c=PAL["dk"]',
    r'fs=6.5,\s*fw="bold",\s*c=PAL\["dk"\]': r'fs=8.5, fw="bold", c=PAL["dk"]',
    r'fs=5.5,\s*c=PAL\["dk"\]': r'fs=7.5, c=PAL["dk"]'
}

for old, new in replacements.items():
    # Only replace if it exactly matches the old pattern
    content = re.sub(old, new, content)

with open('experiments/02_figure/generate_figure2_architecture.py', 'w') as f:
    f.write(content)
