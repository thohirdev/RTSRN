import pickle
import numpy as np
import os

# 1. Buat Matrix Dummy (Misal 37x37 untuk 37 karakter)
# Isinya angka 1 di diagonal (Identity Matrix) agar netral
dummy_matrix = np.eye(68, dtype=np.float32) # Pakai 68 biar aman (biasanya karakter + simbol)

# 2. Pastikan folder loss ada
if not os.path.exists('./loss'):
    os.makedirs('./loss')

# 3. Simpan sebagai confuse.pkl
file_path = './loss/confuse.pkl'
with open(file_path, 'wb') as f:
    pickle.dump(dummy_matrix, f)

print(f"BERHASIL: File dummy '{file_path}' telah dibuat!")