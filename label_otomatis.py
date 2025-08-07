import pandas as pd

# Baca file hasil preprocessing
df = pd.read_excel("tweets_preprocessed.xlsx")

# Daftar kata positif dan negatif sederhana
kata_positif = [
    'bagus', 'baik', 'mendukung', 'hebat', 'senang', 'puas', 'bangga', 'sukses', 'positif',
    'keren', 'berhasil', 'terbaik', 'luar biasa', 'top', 'mantap', 'apresiasi'
]

kata_negatif = [
    'jelek', 'buruk', 'gagal', 'mengecewakan', 'parah', 'marah', 'negatif', 'benci',
    'kecewa', 'tidak setuju', 'tidak puas', 'cacat', 'payah', 'aneh', 'tidak layak'
]
# Fungsi untuk memberi label berdasarkan kata kunci
def label_sentimen(text):
    if not isinstance(text, str):
        return 'netral'
    for word in kata_positif:
        if word in text:
            return 'positif'
    for word in kata_negatif:
        if word in text:
            return 'negatif'
    return 'netral'
# Terapkan fungsi ke setiap tweet
df['label'] = df['preprocessed'].apply(label_sentimen)
# Simpan hasil ke Excel
df.to_excel("tweets_preprocessed.xlsx", index=False)
# Tampilkan ringkasan label
print("Label otomatis selesai ditambahkan.\n")
print("Jumlah data per label:")
print(df['label'].value_counts())


