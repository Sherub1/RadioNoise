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
