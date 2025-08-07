import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, log_loss
from sklearn.preprocessing import LabelEncoder

# === Baca Data Preprocessing ===
df = pd.read_excel("tweets_preprocessed.xlsx")

print("=== Analisis Data Loss ===")
print("Jumlah total data:", len(df))
print("Jumlah tweet kosong (NaN):", df['preprocessed'].isnull().sum())
print("Jumlah label kosong (NaN):", df['label'].isnull().sum())
print("Jumlah tweet yang bersih tapi kosong (''):", (df['preprocessed'].str.strip() == '').sum())

# Hapus data kosong
df = df.dropna(subset=['preprocessed', 'label'])
df = df[df['preprocessed'].str.strip() != '']
print("Jumlah data setelah menghapus data loss:", len(df))
print()

# === Label Encoding (negatif=0, netral=1, positif=2) ===
label_mapping = {'negatif': 0, 'netral': 1, 'positif': 2}
df['label_num'] = df['label'].map(label_mapping)
print("=== Mapping Label ===")
for k, v in label_mapping.items():
    print(f"{k} -> {v}")
print()

# === TF-IDF Vectorization ===
X = df['preprocessed']
y = df['label_num']
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X)

# === Split data ===
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42, stratify=y
)

# === Model Naive Bayes ===
model = MultinomialNB()
model.fit(X_train, y_train)

# === Prediksi ===
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

# === Confusion Matrix ===
print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))
print()

# === Laporan Klasifikasi ===
print("=== Laporan Klasifikasi ===")
print(classification_report(y_test, y_pred, zero_division=0, target_names=['negatif', 'netral', 'positif']))

# === Akurasi ===
acc = accuracy_score(y_test, y_pred)
print("Akurasi:", acc)

# === Log Loss ===
loss = log_loss(y_test, y_prob, labels=[0, 1, 2])
print("Log Loss:", loss)



