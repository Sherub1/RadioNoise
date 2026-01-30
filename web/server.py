#!/usr/bin/env python3
"""
Serveur HTTP local pour l'interface web RadioNoise.
Utilise radionoise.py comme module pour toutes les fonctionnalites.

Usage:
    cd RadioNoise
    python3 web/server.py
"""

import http.server
import socketserver
import json
import time
import sys
import os
import numpy as np
from urllib.parse import urlparse, parse_qs

# Ajouter le dossier parent au path pour importer radionoise
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from radionoise import (
    HardwareRNG,
    NISTTests,
    CHARSETS,
    capture_entropy,
    generate_password,
    is_rtl_sdr_available,
)

# Configuration
PORT = 8742
HOST = "127.0.0.1"

# Liste de mots pour passphrases
WORDLIST = [
    "able", "acid", "aged", "also", "area", "army", "away", "baby",
    "back", "ball", "band", "bank", "base", "bath", "bear", "beat",
    "been", "beer", "bell", "belt", "best", "bill", "bird", "blow",
    "blue", "boat", "body", "bomb", "bond", "bone", "book", "boom",
    "born", "boss", "both", "bowl", "bulk", "burn", "bush", "busy",
    "call", "calm", "came", "camp", "card", "care", "case", "cash",
    "cast", "cell", "chat", "chip", "city", "club", "coal", "coat",
    "code", "cold", "come", "cook", "cool", "cope", "copy", "core",
    "cost", "crew", "crop", "dark", "data", "date", "dawn", "days",
    "dead", "deal", "dean", "dear", "debt", "deep", "deny", "desk",
    "dial", "diet", "dirt", "disc", "disk", "does", "done", "door",
    "dose", "down", "draw", "drew", "drop", "drug", "dual", "duke",
    "dust", "duty", "each", "earn", "ease", "east", "easy", "edge",
    "else", "even", "ever", "evil", "exam", "exit", "face", "fact",
    "fail", "fair", "fall", "fame", "farm", "fast", "fate", "fear",
    "feed", "feel", "feet", "fell", "felt", "file", "fill", "film",
    "find", "fine", "fire", "firm", "fish", "five", "flat", "flow",
    "food", "foot", "ford", "form", "fort", "four", "free", "from",
    "fuel", "full", "fund", "gain", "game", "gate", "gave", "gear",
    "gene", "gift", "girl", "give", "glad", "goal", "goes", "gold",
    "golf", "gone", "good", "gray", "grew", "grey", "grow", "gulf",
    "hair", "half", "hall", "hand", "hang", "hard", "harm", "hate",
    "have", "head", "hear", "heat", "held", "hell", "help", "here",
    "hero", "high", "hill", "hire", "hold", "hole", "holy", "home",
    "hope", "host", "hour", "huge", "hung", "hunt", "hurt", "idea",
    "inch", "into", "iron", "item", "jack", "jane", "jean", "jobs",
    "john", "join", "jump", "jury", "just", "keen", "keep", "kent",
    "kept", "kick", "kill", "kind", "king", "knee", "knew", "know",
    "lack", "lady", "laid", "lake", "land", "lane", "last", "late",
    "lead", "left", "less", "life", "lift", "like", "line", "link",
    "list", "live", "load", "loan", "lock", "logo", "long", "look",
    "lord", "lose", "loss", "lost", "love", "luck", "made", "mail",
]


def get_entropy_source():
    """Determine la source d'entropie active."""
    if is_rtl_sdr_available():
        return "RTL-SDR"
    elif HardwareRNG.is_available():
        return "RDRAND"
    else:
        return "CSPRNG"


def capture_entropy_with_source(samples=500000):
    """Capture l'entropie et retourne (data, source)."""
    source = get_entropy_source()
    entropy = capture_entropy(
        samples=samples,
        allow_fallback=True,
        use_rdseed=False
    )
    return entropy, source


def generate_passphrase_local(random_bytes, words=6):
    """Genere une passphrase avec la wordlist locale."""
    phrase = []
    base = len(WORDLIST)
    threshold = (256 // base) * base
    idx = 0
    while len(phrase) < words and idx < len(random_bytes):
        byte = int(random_bytes[idx])
        idx += 1
        if byte < threshold:
            phrase.append(WORDLIST[byte % base])
    if len(phrase) < words:
        raise ValueError("Not enough entropy")
    return '-'.join(phrase)


class EntropyHandler(http.server.BaseHTTPRequestHandler):
    """Handler HTTP pour l'API d'entropie."""

    def send_cors_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_cors_headers()
        self.end_headers()

    def send_json(self, data, status=200):
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_cors_headers()
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        if path == '/api/status':
            rtl_available = is_rtl_sdr_available()
            rdrand_available = HardwareRNG.is_available()
            entropy_source = get_entropy_source()

            self.send_json({
                'rtl_available': rtl_available,
                'rdrand_available': rdrand_available,
                'entropy_source': entropy_source,
                'server': 'RadioNoise Entropy Server',
                'version': '2.0'
            })

        elif path == '/api/generate':
            try:
                gen_type = params.get('type', ['password'])[0]
                length = int(params.get('length', [16])[0])
                count = int(params.get('count', [1])[0])
                charset = params.get('charset', ['safe'])[0]
                full_test = params.get('full_test', ['false'])[0].lower() == 'true'

                length = max(4, min(128, length))
                count = max(1, min(10, count))

                if charset not in CHARSETS:
                    charset = 'safe'

                # Capture avec fallback automatique
                samples = max(500000, count * length * 100)
                random_bytes, entropy_source = capture_entropy_with_source(samples)

                # Tests NIST (9 rapides ou 15 complets)
                test_size = 50000 if full_test else 10000
                test_data = random_bytes[:min(len(random_bytes), test_size)]
                nist_results = NISTTests.run_all_tests(test_data, verbose=False, fast_mode=not full_test)

                # Generation
                passwords = []
                offset = 0

                for _ in range(count):
                    chunk_size = length * 3 if gen_type == 'password' else length * 2
                    if offset + chunk_size > len(random_bytes):
                        break
                    chunk = random_bytes[offset:offset + chunk_size]
                    offset += chunk_size

                    if gen_type == 'passphrase':
                        pwd = generate_passphrase_local(chunk, words=length)
                        entropy_bits = float(length * np.log2(len(WORDLIST)))
                    else:
                        pwd = generate_password(chunk, length=length, charset=charset)
                        entropy_bits = float(length * np.log2(len(CHARSETS[charset])))

                    passwords.append({
                        'value': pwd,
                        'entropy_bits': round(entropy_bits, 1),
                        'length': len(pwd)
                    })

                # Pipeline description
                if entropy_source == "RTL-SDR":
                    pipeline = ['RTL-SDR', 'Von Neumann', 'SHA-512', 'NIST']
                elif entropy_source in ("RDRAND", "RDSEED"):
                    pipeline = [f'CPU {entropy_source}', 'SHA-512', 'NIST']
                else:
                    pipeline = ['CSPRNG', 'SHA-512', 'NIST']

                self.send_json({
                    'success': True,
                    'passwords': passwords,
                    'nist': {
                        'passed': nist_results['passed'],
                        'total': nist_results['total'],
                        'pass_rate': nist_results['pass_rate']
                    },
                    'entropy_source': entropy_source,
                    'pipeline': pipeline
                })

            except Exception as e:
                self.send_json({
                    'success': False,
                    'error': str(e),
                    'rtl_available': is_rtl_sdr_available(),
                    'rdrand_available': HardwareRNG.is_available()
                }, 500)

        elif path == '/api/test':
            try:
                samples = int(params.get('samples', [100000])[0])
                samples = max(10000, min(1000000, samples))

                entropy, source = capture_entropy_with_source(samples)
                test_data = entropy[:min(len(entropy), 50000)]
                nist_results = NISTTests.run_all_tests(test_data, verbose=False, fast_mode=False)

                self.send_json({
                    'success': True,
                    'entropy_bytes': len(entropy),
                    'entropy_source': source,
                    'nist': {
                        'passed': nist_results['passed'],
                        'total': nist_results['total'],
                        'pass_rate': nist_results['pass_rate'],
                        'tests': [{'name': t['name'], 'passed': t['passed'], 'p_value': t['p_value']}
                                  for t in nist_results['tests']]
                    }
                })

            except Exception as e:
                self.send_json({
                    'success': False,
                    'error': str(e)
                }, 500)

        else:
            self.send_json({'error': 'Not found'}, 404)

    def log_message(self, format, *args):
        print(f"[{time.strftime('%H:%M:%S')}] {args[0]}")


def main():
    print("=" * 60)
    print("RadioNoise Entropy Server v2.0")
    print("=" * 60)

    # Verifier les sources d'entropie
    print("\nSources d'entropie:")

    rtl_ok = is_rtl_sdr_available()
    rdrand_ok = HardwareRNG.is_available()

    if rtl_ok:
        print("  [1] RTL-SDR        : disponible")
    else:
        print("  [1] RTL-SDR        : non detecte")

    if rdrand_ok:
        print("  [2] CPU RDRAND     : disponible")
    else:
        print("  [2] CPU RDRAND     : non disponible")

    print("  [3] System CSPRNG  : disponible (fallback)")

    # Source active
    source = get_entropy_source()
    print(f"\n-> Source active: {source}")

    print(f"\nServeur: http://{HOST}:{PORT}")
    print(f"Interface: ouvrir web/index.html dans un navigateur")
    print("\nEndpoints:")
    print("  GET /api/status   - Etat du serveur")
    print("  GET /api/generate - Generer des mots de passe")
    print("  GET /api/test     - Tests NIST complets")
    print("\nCtrl+C pour arreter\n")

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer((HOST, PORT), EntropyHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nArret du serveur...")


if __name__ == "__main__":
    main()
