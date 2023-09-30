# nqa
# Import các thư viện cần thiết
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Đọc dữ liệu từ file CSV
data = pd.read_csv('data.csv')

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(data['Bình luận'], data['Nhãn cảm xúc'], test_size=0.2, random_state=42)

# Tiền xử lý dữ liệu văn bản
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Xây dựng mô hình Logistic Regression
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Dự đoán nhãn cảm xúc trên tập kiểm tra
y_pred = model.predict(X_test_vec)

# Đánh giá độ chính xác của mô hình
accuracy = accuracy_score(y_test, y_pred)
print("Độ chính xác:", accuracy)
