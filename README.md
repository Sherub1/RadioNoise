# RadioNoise : Générer du vrai hasard avec une clé USB à 30€

## Introduction : Le problème du "faux" hasard

Imaginez que vous lancez un dé. Le résultat semble parfaitement aléatoire, n'est-ce pas ? Pourtant, si vous connaissiez exactement la force de votre lancer, l'angle initial, la friction de la table... vous pourriez prédire le résultat. Ce n'est pas du vrai hasard, c'est juste de la complexité.

C'est exactement le problème avec les ordinateurs. Quand votre ordinateur génère un "nombre aléatoire", il utilise en réalité des formules mathématiques complexes. Ces **générateurs pseudo-aléatoires** (PRNG) produisent des séquences qui *semblent* aléatoires, mais qui sont en fait complètement prévisibles si on connaît la "graine" de départ. C'est comme un dé pipé ultra-sophistiqué.

Pour la plupart des usages (jeux vidéo, simulations, etc.), ce n'est pas un problème. Mais pour la **cryptographie**, c'est catastrophique. Vos clés de chiffrement, vos certificats SSL, vos portefeuilles de cryptomonnaies... tout repose sur du vrai hasard. Un seul nombre prévisible et c'est toute votre sécurité qui s'effondre.

## La solution : capturer le chaos de l'univers

Pour obtenir du *vrai* hasard, il faut sortir du monde déterministe de l'ordinateur et aller chercher dans la nature. Et la nature est fondamentalement chaotique au niveau quantique. Les fluctuations électromagnétiques, le bruit thermique, les rayons cosmiques... autant de phénomènes impossibles à prédire.

**RadioNoise** est un projet qui exploite ce chaos naturel pour générer des nombres véritablement aléatoires. Et la source d'entropie utilisée ? Le **bruit radio capturé par une simple clé USB RTL-SDR**.

## Qu'est-ce qu'un RTL-SDR ?

Un RTL-SDR (Software Defined Radio) est une petite clé USB, semblable à une clé TNT, qui coûte entre 25 et 40 euros. À l'origine conçue pour recevoir la télévision numérique, la communauté des radio-amateurs a découvert qu'elle pouvait être détournée pour capter une très large gamme de fréquences radio : de 24 MHz à 1,7 GHz.

Avec ce petit appareil, on peut écouter :
- Les communications aériennes
- Les balises météo
- Les satellites en orbite
- Les signaux GPS
- La radio FM
- Et... le **bruit de fond radio de l'univers**

C'est ce dernier point qui nous intéresse.

## Le bruit radio : une source d'entropie pratique

Quand vous allumez une radio FM et que vous vous mettez entre deux stations, vous entendez un grésillement. Ce "bruit blanc" n'est pas du silence, c'est un véritable océan de signaux aléatoires :

- **Bruit thermique (Johnson-Nyquist)** : L'agitation thermique des électrons dans les circuits électroniques produit des fluctuations de tension imprévisibles. C'est un phénomène de physique statistique classique.
- **Bruit atmosphérique** : Les éclairs, les orages, les perturbations ionosphériques génèrent des ondes radio chaotiques
- **Bruit galactique** : Les étoiles, les pulsars, les supernovae émettent en permanence un rayonnement électromagnétique
- **Bruit de grenaille (shot noise)** : Dans les composants électroniques du récepteur, le passage discret des électrons crée des fluctuations. Ce phénomène a une origine quantique.

**Note importante** : Contrairement à ce qu'on pourrait lire ailleurs, le bruit capté par un RTL-SDR n'est pas majoritairement "quantique". Le bruit thermique, qui domine, est un phénomène de physique statistique classique décrit par la distribution de Boltzmann. Il devient quantique uniquement à très basse température (quand ℏω > kT).

Cela dit, ce bruit reste **pratiquement imprévisible** : la complexité des interactions thermiques et électromagnétiques rend toute prédiction impossible en pratique. C'est suffisant pour générer de l'entropie de qualité cryptographique, même si ce n'est pas du "vrai" hasard quantique au sens strict.

## Comment ça marche ?

Le processus de QuantumNoise se déroule en plusieurs étapes :

### 1. Capture du signal radio

Le RTL-SDR se branche sur un port USB et se met à l'écoute d'une fréquence spécifique (par exemple 100 MHz). À cette fréquence, il n'y a généralement aucune émission radio, juste du bruit de fond. Le récepteur échantillonne ce bruit plusieurs millions de fois par seconde et renvoie un flux de données brutes.

```python
# Initialisation du récepteur RTL-SDR
sdr = RtlSdr()
sdr.sample_rate = 2.4e6  # 2.4 millions d'échantillons/seconde
sdr.center_freq = 100e6  # Fréquence à 100 MHz
sdr.gain = 'auto'

# Capture de 256k échantillons de bruit
samples = sdr.read_samples(256 * 1024)
```

### 2. Conversion en données numériques

Les échantillons capturés sont des nombres complexes représentant l'amplitude et la phase du signal. On les convertit en valeurs réelles (par exemple, en prenant le module ou la partie réelle).

```python
# Extraction de l'amplitude du signal
amplitude = np.abs(samples)
```

### 3. Extraction de l'entropie

Le signal brut n'est pas parfaitement uniforme. Certaines fréquences peuvent être légèrement plus fortes, il peut y avoir des biais dus au hardware. Pour extraire de l'entropie pure, on applique plusieurs techniques :

- **Échantillonnage des bits de poids faible** : On ne garde que les bits les moins significatifs de chaque échantillon, là où le bruit quantique domine
- **XOR entre échantillons successifs** : On combine les échantillons entre eux pour éliminer les corrélations temporelles
- **Hachage cryptographique** : On passe le flux dans une fonction de hachage (SHA-256) pour uniformiser la distribution

```python
# On ne garde que les 2 bits de poids faible
low_bits = (amplitude * 4).astype(int) & 0b11

# Conversion en octets
random_bytes = np.packbits(low_bits)
```

### 4. Post-traitement et "blanchiment"

Même après extraction, les données peuvent présenter de légères corrélations. On applique alors un algorithme de **whitening** (blanchiment) qui garantit une distribution parfaitement uniforme :

```python
# Passage dans un extracteur de randomness (par exemple, SHA-256)
import hashlib
whitened = hashlib.sha256(random_bytes).digest()
```

### 5. Validation de la qualité

Avant d'utiliser ces nombres aléatoires en cryptographie, il faut s'assurer qu'ils sont vraiment imprévisibles. On les soumet à des batteries de tests statistiques :

- **Tests NIST** : 15 tests différents qui vérifient l'absence de patterns
- **Test d'entropie de Shannon** : Mesure du contenu informationnel (doit être proche de 8 bits/octet)
- **Tests de compression** : Un bon aléatoire est incompressible

Si les données passent tous ces tests, on peut être certain qu'elles sont cryptographiquement sûres.

## Pourquoi c'est mieux qu'un PRNG ?

Comparons les différentes approches :

### Générateur Pseudo-Aléatoire (PRNG)
```
Graine (seed) → Formule mathématique → Séquence "aléatoire"
```
- ✅ Très rapide
- ✅ Reproductible (utile pour les tests)
- ❌ Prévisible si on connaît la graine
- ❌ Sécurité dépend de la qualité de la graine initiale

### Générateur Matériel (TRNG - True Random Number Generator)
```
Phénomène physique (bruit thermique, atmosphérique) → Capture → Hasard pratique
```
- ✅ Imprévisible en pratique (complexité chaotique)
- ✅ Source d'entropie continue
- ✅ Indépendant de l'état logiciel du système
- ❌ Plus lent (limité par le hardware)
- ❌ Nécessite du matériel supplémentaire
- ⚠️ Théoriquement déterministe (physique classique)

**C'est ce que fait RadioNoise** avec le RTL-SDR.

### Générateur Quantique (QRNG)
```
Phénomène quantique (photons, shot noise) → Mesure → Vrai hasard fondamental
```
- ✅ Fondamentalement indéterministe (mécanique quantique)
- ✅ Impossible à prédire même avec une connaissance parfaite du système
- ❌ Matériel spécialisé coûteux (diodes Zener, optique quantique)
- ❌ Plus complexe à mettre en œuvre correctement

Pour la plupart des usages cryptographiques, un **TRNG bien conçu comme RadioNoise est largement suffisant**. Les QRNG sont utiles pour les applications où l'on veut une garantie théorique absolue, mais en pratique, un attaquant ne peut pas plus prédire le bruit thermique que le résultat d'une mesure quantique.

## Applications pratiques

À quoi peut servir RadioNoise concrètement ?

### Génération de clés cryptographiques
```python
# Générer une clé AES-256 (32 octets)
key = radionoise.generate(32)
```

### Création de tokens de sécurité
```python
# Token de session imprévisible
token = radionoise.generate(64).hex()
```

### Initialisation de wallets crypto
```python
# Graine pour un wallet Bitcoin/Ethereum
seed = radionoise.generate(32)
```

### Amélioration de /dev/random sous Linux
Les systèmes Linux modernes collectent de l'entropie depuis diverses sources (mouvements de souris, timings des disques, etc.). RadioNoise pourrait alimenter directement le pool d'entropie système pour renforcer la sécurité globale.

## Les défis techniques

Capturer du bruit radio semble simple, mais plusieurs pièges guettent :

### 1. Les interférences
Si une station radio émet sur la fréquence choisie, le signal "pollue" le bruit naturel. Il faut donc choisir des fréquences vraiment désertes, ou filtrer les émissions.

### 2. Les biais du hardware
Chaque RTL-SDR a ses propres défauts : gain non-linéaire, bruit thermique du circuit, etc. Ces biais doivent être caractérisés et compensés.

### 3. La vitesse de génération
Un RTL-SDR échantillonne à ~2.4 MHz, mais après extraction et blanchiment, le débit d'entropie utilisable tombe à quelques dizaines de ko/s. C'est suffisant pour générer des clés, mais pas pour chiffrer un flux vidéo en temps réel.

### 4. La sécurité physique
Si un attaquant peut émettre un signal radio puissant vers votre antenne RTL-SDR, il pourrait théoriquement "forcer" le générateur à produire des valeurs prévisibles. Une cage de Faraday et un monitoring actif sont nécessaires en environnement hostile.

## Pourquoi ce projet ?

RadioNoise est né d'une question simple : *"Peut-on créer un vrai générateur d'entropie avec du matériel accessible ?"*

Les solutions commerciales (TrueRNG, Entropy Key) coûtent plusieurs centaines d'euros et sont des boîtes noires. Les circuits à diode Zener nécessitent des compétences en électronique et un ADC de qualité. 

Le RTL-SDR, lui, est :
- **Abordable** : 25-40€ sur Amazon/AliExpress
- **Accessible** : Simple clé USB, plug-and-play
- **Documenté** : Communauté active, drivers open-source
- **Polyvalent** : Peut aussi servir pour la radio amateur !

C'est la solution idéale pour expérimenter avec la génération d'entropie matérielle sans investissement lourd.

## Conclusion

Dans un monde où la sécurité informatique repose sur la qualité du hasard, disposer d'une source d'entropie fiable est crucial. RadioNoise démontre qu'il n'est pas nécessaire d'acheter du matériel coûteux ou de construire des circuits complexes pour y parvenir.

Avec une simple clé USB à 30€ et quelques lignes de Python, on peut capter le bruit électromagnétique ambiant et le transformer en nombres pratiquement imprévisibles. C'est de la cryptographie DIY dans ce qu'elle a de plus élégant : simple, accessible, et fondée sur la physique du chaos thermique.

**Soyons honnêtes** : ce n'est pas un générateur "quantique" au sens strict. Le bruit thermique est un phénomène classique. Mais en pratique, c'est tout aussi imprévisible : personne ne peut prédire l'agitation de milliards d'électrons dans un circuit. Pour la cryptographie du quotidien, c'est largement suffisant.

Le projet est encore en développement, mais les bases sont posées. L'objectif à terme est de créer un outil libre, auditable, et utilisable par tous ceux qui ont besoin de vrai hasard.

---

**RadioNoise** - *Capturer le chaos, générer la confiance*
