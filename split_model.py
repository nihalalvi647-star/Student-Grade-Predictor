import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 1. डेटा लोड करना
data = pd.read_csv('Student_data.csv')

# 2. Features (X) और Target (y) को अलग करना
X = data.drop(columns=['Final_Sgrade']) 
y = data['Final_Sgrade']

# 3. [नया कॉन्सेप्ट] डेटा को Train (80%) और Test (20%) में बांटना
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.metrics import mean_absolute_error

# 4. मॉडल को सिर्फ ट्रेनिंग डेटा पर सिखाना (Train करना)
model = LinearRegression()
model.fit(X_train, y_train)

print("--- 🎯 बधाई हो! नया स्प्लिट मॉडल तैयार है ---")
print(f"ट्रेनिंग के लिए इस्तेमाल हुआ डेटा (Rows): {len(X_train)}")
print(f"टेस्टिंग के लिए बचा हुआ डेटा (Rows): {len(X_test)}")
# 5. छुपाए हुए टेस्ट डेटा पर भविष्यवाणी करना
y_pred = model.predict(X_test)

# 6. मॉडल की गलती (Mean Absolute Error) नापना
mae = mean_absolute_error(y_test, y_pred)

print("\n--- 📊 मॉडल का लाइव टेस्ट रिजल्ट ---")
print(f"असली मार्क्स (Actual Grade): {y_test.values[0]:.2f}%")
print(f"मॉडल की भविष्यवाणी (Predicted Grade): {y_pred[0]:.2f}%")
print(f"औसत गलती (Mean Absolute Error): {mae:.2f}%")
