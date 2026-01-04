# BanditGPT Priors Directory

This directory contains symlinks to the canonical offline dataset files in `../data/offline_dataset/`.

## Files

- **`priors_meta_pca.npz`** → `../data/offline_dataset/priors_meta_pca.npz`
  - Covariance matrix and cluster statistics
  - Built from 21,719 offline prompts (all - train - test)
  - See `../data/offline_dataset/README.md` for details

## Why Symlinks?

The priors are now maintained in the `offline_dataset/` folder for better organization and reproducibility. This directory contains symlinks for backward compatibility with existing code that expects priors in `banditgpt/priors/`.

## Canonical Location

For the authoritative priors data and generation scripts, see:
```
banditgpt/data/offline_dataset/
├── priors_meta_pca.npz          # Canonical file
├── generate_dataset.py          # Generation script
└── README.md                    # Full documentation
```
