import pandas as pd 
# 1. CSV file ko load karna 
data = pd.read_csv('Student_data.csv')
# 2.data ko screen per print karke dekhna
print("--- we first dataset --")
print(data)
# 3.(Rows, Columns, Data types)
print("\n--- data ki jankari ---")
print(data.info())

print("\n--- (Summary) ---")
print(data.describe())
print("\n--- डेटा की बुनियादी जानकारी ---")
print(data.info())
print("\n--- डेटा का गणितीय विश्लेषण (Summary) ---")
print(data.describe())
X = data.drop(columns=['Final_Sgrade']) 
y = data['Final_Sgrade']
print("\n--- हमारे Features (X) ---")
print(X.head()) # यह X की शुरुआती लाइनें दिखाएगा
print("\n--- हमारा Target (y) ---")
print(y.head()) # यह y की शुरुआती लाइनें दिखाएगा
# 6. मशीन लर्निंग लाइब्रेरी को इम्पोर्ट करना
from sklearn.linear_model import LinearRegression

# 7. मॉडल को बनाना और ट्रेन (Train) करना
model = LinearRegression()
model.fit(X, y) # यहाँ हमारा लैपटॉप डेटा देखकर सीख रहा है

print("\n--- बधाई हो! आपका मॉडल ट्रेन हो गया है ---")
# 8. नए छात्र के डेटा पर भविष्यवाणी करना
# फॉर्मेट: [[Hours_Studied, Sleep_Hours, Previous_Score]]
new_student = [[6, 7, 80]] 

predicted_grade = model.predict(new_student)

print("\n--- 🎯 मशीन लर्निंग की भविष्यवाणी 🎯 ---")
print(f"अगर छात्र 6 घंटे पढ़ेगा, तो उसके संभावित मार्क्स होंगे: {predicted_grade[0]:.2f}%")
import matplotlib.pyplot as plt
# 9. डेटा विज़ुअलाइज़ेशन (ग्राफ़ बनाना)
print("\n--- 📊 आपका पहला डेटा ग्राफ़ तैयार हो रहा है... ---")

# पढ़ाई के घंटों और फाइनल मार्क्स के बीच का संबंध दिखाना
plt.scatter(data['Hours_Studied'], data['Final_Sgrade'], color='blue', marker='o')

# ग्राफ़ पर नाम लिखना (Labels)
plt.title('Hours Studied vs Final Grade')
plt.xlabel('Hours of Study')
plt.ylabel('Final Grade (%)')
plt.grid(True) # पीछे ग्रिड लाइन्स दिखाना

# ग्राफ़ को स्क्रीन पर दिखाना
plt.show()
# ग्राफ़ को लैपटॉप में इमेज के रूप में सेव करना
plt.savefig('student_performance_graph.png')
