# RadioNoise : Générer du vrai hasard avec une clé RTL-SDR

## Introduction

Les générateurs pseudo-aléatoires (PRNG) utilisés par les ordinateurs produisent des séquences déterministes. Pour les applications cryptographiques, cette prévisibilité constitue une vulnérabilité. Une alternative consiste à exploiter des phénomènes physiques comme source d'entropie.

RadioNoise utilise le bruit électromagnétique capté par un récepteur RTL-SDR comme source de hasard. Ce bruit résulte de la superposition de plusieurs phénomènes :

- **Bruit thermique (Johnson-Nyquist)** : fluctuations de tension causées par l'agitation thermique des porteurs de charge dans les conducteurs
- **Bruit atmosphérique** : perturbations électromagnétiques d'origine météorologique
- **Bruit galactique** : rayonnement de fond d'origine cosmique
- **Bruit de grenaille** : fluctuations liées au caractère discret des charges électriques

## Méthode d'extraction

Le signal brut capturé présente des biais statistiques liés aux caractéristiques du récepteur. L'extracteur de Von Neumann corrige ce problème en analysant les bits par paires :

- Si deux bits consécutifs diffèrent (01 ou 10), le premier bit est conservé
- Si deux bits sont identiques (00 ou 11), la paire est rejetée

Cette méthode garantit une distribution uniforme en sortie, indépendamment du biais initial, au prix d'une réduction du débit (efficacité théorique de 25%, environ 3% en pratique avec les corrélations du signal).

## Validation expérimentale

Une capture réalisée le 27 janvier 2025 illustre le processus :

| Paramètre | Valeur |
|-----------|--------|
| Fréquence centrale | 100 MHz |
| Taux d'échantillonnage | 2.4 MS/s |
| Volume capturé | 3,500 Ko |
| Volume extrait | 109 Ko |
| Efficacité | 3.1% |

L'analyse des données extraites confirme leur qualité :

| Mesure | Résultat |
|--------|----------|
| Entropie de Shannon | 7.9986 bits/octet |
| Ratio bits 0/1 | 49.99% / 50.01% |
| Autocorrélation | < 0.002 |
| Tests NIST SP 800-22 | 7/7 réussis |

Ces résultats attestent d'une qualité suffisante pour un usage cryptographique. Les données extraites sont statistiquement indiscernables d'une source parfaitement aléatoire.

## Limites

Le bruit thermique, qui constitue la composante dominante du signal capturé, relève de la physique statistique classique et non de la mécanique quantique. L'imprévisibilité obtenue est donc pratique plutôt que fondamentale. Néanmoins, la complexité des interactions thermiques rend toute prédiction computationnellement impossible, ce qui suffit pour les applications cryptographiques courantes.

## Générateur de mots de passe

### Pipeline de traitement

```
[Source Entropie] → [Extraction Von Neumann] → [Whitening SHA-512] → [Tests NIST] → [Mots de passe]
```

### Sources d'entropie (par priorité)

1. **RTL-SDR** - Bruit radio (thermique + atmosphérique) capturé via dongle USB
2. **RDRAND** - RNG matériel CPU Intel/AMD (DRBG interne)
3. **RDSEED** - Entropie directe CPU (plus lent, plus pur)
4. **CSPRNG** - `secrets.token_bytes()` (fallback ultime)

Le script bascule automatiquement vers la source suivante si la précédente n'est pas disponible.

### Tests NIST implémentés

| Mode | Tests | Temps |
|------|-------|-------|
| Rapide (défaut) | 9 tests essentiels | ~1-2s |
| Complet (`--full-test`) | 15 tests SP 800-22 | ~30s |

Tests inclus : Frequency, Block Frequency, Runs, Longest Run, Spectral (DFT), Serial, Approximate Entropy, Cumulative Sums, Binary Matrix Rank, Template Matching, Maurer's Universal, Linear Complexity, Random Excursions.

### Utilisation

```bash
# Génération standard (5 mots de passe, 16 caractères)
python3 radionoise.py

# Personnaliser longueur et nombre
python3 radionoise.py -l 20 -n 10

# Passphrases (mots)
python3 radionoise.py -p -l 6

# Suite complète NIST (plus lent)
python3 radionoise.py --full-test

# Test d'un fichier d'entropie existant
python3 radionoise.py -f entropy.bin --test-only

# Forcer RDSEED au lieu de RDRAND
python3 radionoise.py --use-rdseed
```

### Options principales

| Option | Description |
|--------|-------------|
| `-l, --length` | Longueur du mot de passe (défaut: 16) |
| `-n, --count` | Nombre de mots de passe (défaut: 5) |
| `-c, --charset` | Jeu de caractères: `alnum`, `alpha`, `digits`, `hex`, `full`, `safe` |
| `-p, --passphrase` | Génère des passphrases (mots séparés par `-`) |
| `-f, --file` | Utilise un fichier d'entropie existant |
| `--full-test` | Suite complète des 15 tests NIST |
| `--no-test` | Désactive la validation NIST |
| `--use-rdseed` | Utilise RDSEED au lieu de RDRAND |
| `-q, --quiet` | Mode silencieux |

### Dépendances

- Python 3 avec `numpy`, `scipy`
- GCC (compilation automatique du module RDRAND)
- `rtl_sdr` (optionnel, pour source RTL-SDR)

### Sécurité

- **Effacement mémoire** : `secure_zero()` tente d'effacer les données sensibles (best-effort en Python)
- **Rejection sampling** : Évite les biais de modulo lors de la conversion bytes→caractères
- **Seuil NIST** : p-value >= 0.01 (confiance 99%)
