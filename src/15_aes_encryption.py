
import os
import json
import time
import statistics
from pathlib import Path

import pandas as pd
from cryptography.hazmat.primitives.ciphers.aead import (
    AESGCM,
    ChaCha20Poly1305,
)
from cryptography.exceptions import InvalidTag

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

# Demo patient data (no real patient details)
patient = {
    "Age": 12,
    "BMI": 18.5,
    "WBC_Count": 11.2,
    "Diagnosis": "appendicitis"
}

plaintext = json.dumps(patient).encode("utf-8")


def encrypt_data(data, key, algorithm):
    cipher = algorithm(key)
    nonce = os.urandom(12)
    ciphertext = cipher.encrypt(nonce, data, None)
    return nonce, ciphertext


def decrypt_data(nonce, ciphertext, key, algorithm):
    cipher = algorithm(key)
    return cipher.decrypt(nonce, ciphertext, None)


algorithms = {
    "AES-256-GCM": AESGCM,
    "ChaCha20-Poly1305": ChaCha20Poly1305,
}

results = []
RUNS = 1000

print("=== FYP SECURITY COMPARISON ===")

for name, algorithm in algorithms.items():
    key = algorithm.generate_key(bit_length=256) if name == "AES-256-GCM" else algorithm.generate_key()

    # Warm-up
    for _ in range(50):
        n, c = encrypt_data(plaintext, key, algorithm)
        decrypt_data(n, c, key, algorithm)

    encryption_times = []
    decryption_times = []

    for _ in range(RUNS):
        start = time.perf_counter_ns()
        nonce, ciphertext = encrypt_data(plaintext, key, algorithm)
        encryption_times.append(
            (time.perf_counter_ns() - start) / 1_000_000
        )

        start = time.perf_counter_ns()
        decrypted = decrypt_data(
            nonce, ciphertext, key, algorithm
        )
        decryption_times.append(
            (time.perf_counter_ns() - start) / 1_000_000
        )

        assert decrypted == plaintext

    # Verify authentication / tampering detection
    nonce, ciphertext = encrypt_data(plaintext, key, algorithm)
    tampered = bytearray(ciphertext)
    tampered[0] ^= 1

    tamper_detected = False
    try:
        decrypt_data(nonce, bytes(tampered), key, algorithm)
    except InvalidTag:
        tamper_detected = True

    result = {
        "Algorithm": name,
        "Key_Bits": 256,
        "Encryption_ms": round(statistics.mean(encryption_times), 6),
        "Decryption_ms": round(statistics.mean(decryption_times), 6),
        "Integrity_Verified": tamper_detected,
    }

    results.append(result)

    print("\nAlgorithm:", name)
    print("Encryption time:", result["Encryption_ms"], "ms")
    print("Decryption time:", result["Decryption_ms"], "ms")
    print("Integrity verified:", tamper_detected)

df = pd.DataFrame(results)

file = OUT / "aes_chacha_comparison.csv"
df.to_csv(file, index=False)

print("\n=== FINAL COMPARISON ===")
print(df.to_string(index=False))
print("\nSaved:", file)
