import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv('Driver_Drowsiness_3000.csv')
df.columns = ['Eye Aspect Ratio', 'Mouth Aspect Ratio', 'target']

print(df.head())

X = df[['Eye Aspect Ratio' , 'Mouth Aspect Ratio']]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

modelo = LogisticRegression()
modelo.fit(X_train, y_train)

#revisão de 20% de dados escondidos
y_pred = modelo.predict(X_test)

#avaliar acurácia
acuracia = accuracy_score(y_test, y_pred)
print(f"Acurácia do modelo: {acuracia * 100:.2f}%\n")

print("Relatório de Classificação:")
print(classification_report(y_test, y_pred))