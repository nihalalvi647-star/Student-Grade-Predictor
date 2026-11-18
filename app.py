import pandas as pd
from sklearn.linear_model import LinearRegression

# 1. पुराना डेटा लोड करके मॉडल को ट्रेन करना
data = pd.read_csv('Student_data.csv')
X = data.drop(columns=['Final_Sgrade']) 
y = data['Final_Sgrade']

model = LinearRegression()
model.fit(X, y)

print("=== 🎓 STUDENT PERFORMANCE PREDICTOR SYSTEM 🎓 ===")
print("कृपया छात्र की जानकारी दर्ज करें:\n")

# 2. यूजर से लाइव इनपुट लेना
studying_hours = float(input("1. रोज़ाना पढ़ाई के घंटे (Hours): "))
sleeping_hours = float(input("2. रोज़ाना सोने के घंटे (Hours): "))
past_score = float(input("3. पिछले एग्जाम के मार्क्स (%): "))

# 3. इनपुट को मॉडल के फॉर्मेट में बदलना
user_data = [[studying_hours, sleeping_hours, past_score]]

# 4. भविष्यवाणी करना
predicted_grade = model.predict(user_data)

print("\n------------------------------------------------")
print(f"🎯 मशीन लर्निंग का अनुमान: छात्र के संभावित मार्क्स होंगे: {predicted_grade[0]:.2f}%")
print("------------------------------------------------")
