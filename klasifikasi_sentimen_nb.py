import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# Baca file hasil preprocessing
df = pd.read_excel("tweets_preprocessed.xlsx")

# Analisis data loss
print("=== Analisis Data Loss ===")
print("Jumlah total data:", len(df))
print("Jumlah tweet kosong (NaN):", df['preprocessed'].isnull().sum())
print("Jumlah label kosong (NaN):", df['label'].isnull().sum())
print("Jumlah tweet yang bersih tapi kosong (''):", (df['preprocessed'].str.strip() == '').sum())

# Hapus baris yang mengandung data kosong
df = df.dropna(subset=['preprocessed', 'label'])
df = df[df['preprocessed'].str.strip() != '']
print("Jumlah data setelah menghapus data loss:", len(df))
print()

# Siapkan fitur dan label
X = df['preprocessed']
y = df['label']

# Konversi teks ke fitur TF-IDF
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X)

# Split data: 80% latih, 20% uji, dengan stratifikasi label
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42, stratify=y
)

# Inisialisasi model Naive Bayes
model = MultinomialNB()

# Latih model
model.fit(X_train, y_train)

# Prediksi data uji
y_pred = model.predict(X_test)

# Evaluasi model
print("=== Laporan Klasifikasi ===")
print(classification_report(y_test, y_pred, zero_division=0))
print("Akurasi:", accuracy_score(y_test, y_pred))


