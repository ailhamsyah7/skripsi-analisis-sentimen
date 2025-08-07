import pandas as pd
import re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Baca file Excel
df = pd.read_excel("tweets.xlsx")
# Cek nama kolom
print("Kolom dalam file:", df.columns)
# Inisialisasi stopword dan stemmer
stop_factory = StopWordRemoverFactory()
stopword = stop_factory.create_stop_word_remover()
stem_factory = StemmerFactory()
stemmer = stem_factory.create_stemmer()
# Fungsi membersihkan teks
def clean_text(text):
    if pd.isnull(text):
        return ""
    text = str(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'@\w+|\#', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = stopword.remove(text)
    text = stemmer.stem(text)
    return text
# Ganti 'Tweet' jika kolom bukan 'tweet'
df['preprocessed'] = df['tweet'].apply(clean_text)
# Simpan ke file baru
df.to_excel("tweets_preprocessed.xlsx", index=False)
print("Preprocessing selesai. Disimpan ke 'tweets_preprocessed.xlsx'")
