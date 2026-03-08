# Enigma-Cracker-with-CUDA-Gillogly-Attack-
This project implements an Enigma cipher cracking tool using the Gillogly attack (a known-plaintext‑less method) and CUDA acceleration. It is designed to run on a single GPU (tested on RTX6000 Ada) and can recover the rotor settings and plugboard configuration from a ciphertext of sufficient length.
---

📌 Features

· Two‑phase attack
  1. GPU phase – brute‑forces all rotor orders (6) and initial positions (26³) using the Index of Coincidence to filter candidates.
  2. CPU phase – for each promising rotor setting, performs a hill‑climbing search on the plugboard using trigram statistics.
· Fully parallelised on the GPU – processes 105 456 rotor settings in seconds.
· CUDA constant memory used for rotor wirings and frequency tables for maximum performance.
· Stand‑alone – no external libraries except the CUDA toolkit.

---

⚙️ Requirements

· GPU with CUDA Compute Capability 8.9 (e.g. RTX6000 Ada) – can be adapted to other architectures by changing -arch
· CUDA Toolkit 11.0 or newer
· C++11 compatible compiler (g++ / MSVC)

---

🚀 Compilation

Save the source code as enigma_cracker.cu and compile with:

```bash
nvcc -o enigma_cracker enigma_cracker.cu -arch=sm_89 -O3 -std=c++11
```

If your GPU has a different compute capability, replace sm_89 with the appropriate value (e.g. sm_80 for Ampere).
You can check your GPU’s compute capability with deviceQuery (part of the CUDA samples).

---

🧪 Usage

1. Prepare the ciphertext

Two options:

· Edit the source – replace the placeholder string in main():
  ```cpp
  const char* ciphertext = "YOUR_ENIGMA_CIPHERTEXT_HERE";
  ```
  The ciphertext must contain only uppercase letters A‑Z (no spaces, no punctuation).
· Read from a file – uncomment the file‑reading block in main() and create a cipher.txt file containing the ciphertext.

2. Run the program

```bash
./enigma_cracker
```

3. Interpret the output

· Phase 1 prints the top rotor candidates together with their Index of Coincidence (IC).
  Values close to 0.076 indicate a likely correct rotor setting (German language).
· Phase 2 tries to recover the plugboard for each promising candidate.
  The final decrypted message is shown at the end.

Note: For reliable results the ciphertext should be at least 150 characters long. Shorter texts may still work but the success rate drops.

---

🔧 Customisation

· Language model – the program currently uses German letter frequencies and trigrams.
  To crack messages in another language, replace d_german_freq and the common_trigrams array in trigram_score() with appropriate statistics.
· Rotor wirings – if your Enigma variant uses different rotors, update the h_rotor_wiring arrays accordingly.
· GPU selection – the code uses device 0 by default. Change cudaSetDevice(0) to select another GPU.

---

📄 License

This code is provided for educational and research purposes only. You are free to use, modify and distribute it, but please retain the original attribution.

---

🤝 Contributing

Feel free to open issues or submit pull requests if you find bugs or have improvements (e.g. multi‑GPU support, better trigram scoring, other languages).

---

Happy cracking!
