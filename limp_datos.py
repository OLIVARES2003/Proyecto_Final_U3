import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Cargar los datos y eliminar espacios en los nombres de las columnas
file_path = 'salud_mental.xlsx'
df = pd.read_excel(file_path, sheet_name='Respuestas de formulario 1')
df.columns = df.columns.str.strip()

# Separar datos numéricos y textuales
numerical_cols = [
    "¿Cuántas horas duermes en promedio cada noche?  (Opciones 1 hr o 5 hrs)",	 
      "¿Con qué frecuencia sientes estrés debido a la carga académica?",   
            	  "¿Cuántas veces al mes participas en actividades de relajación o deportivas para reducir el estrés?", 	
                      "¿Cuántas horas dedicas semanalmente a actividades recreativas?   (Opciones 1 hr o 5 hrs)",	
                          "¿Qué nivel de satisfacción tienes con tu vida académica actualmente? (Escala de 1 a 10)"  

]
text_cols = [
    "¿Cómo describirías el impacto del estrés académico en tu vida cotidiana?",
    "¿Qué estrategias utilizas para gestionar el estrés?",
    "¿Qué sientes que es lo más desafiante en mantener un equilibrio entre estudios y salud mental?",
    "¿Cómo influye tu vida académica en tu bienestar emocional?",
    "¿Qué cambios o recursos en la universidad crees que ayudarían a mejorar la salud mental de los estudiantes?", 	
    "¿Cómo te sientes con respecto a tu carga académica y su impacto en tu salud mental?"

]

numerical_data = df[numerical_cols].dropna()
text_data = df[text_cols].dropna()

# Archivo Excel
output_path = 'datos.xlsx'
with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    numerical_data.to_excel(writer, sheet_name='Datos Numéricos', index=False)
    text_data.to_excel(writer, sheet_name='Datos Textuales', index=False)

print(f"Datos exportados a '{output_path}'.")


# Estadísticas descriptivas
print("\nEstadísticas descriptivas:")
print(numerical_data.describe())

# Visualización de histogramas
numerical_data.hist(bins=10, figsize=(10, 8))
plt.suptitle("Distribución de datos numéricos")
plt.show()

# Diagramas de dispersión
sns.pairplot(numerical_data)
plt.show()

# Nube de palabras
all_text = " ".join(text_data[text_cols[0]].dropna())
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Nube de palabras: Impacto de la carga académica")
plt.show()

# Estandarización de datos
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numerical_data)

# Determinación del número óptimo de clústeres
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia, marker='o')
plt.title("Método del codo")
plt.xlabel("Número de clústeres")
plt.ylabel("Inercia")
plt.show()

#K-means
optimal_k = 2  # Ajusta esto según el gráfico del codo
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(scaled_data)


numerical_data["Cluster"] = clusters

# Visualización de clústeres
sns.pairplot(numerical_data, hue="Cluster", palette="tab10")
plt.show()


# Preparar datos para clasificación
X = scaled_data
y = clusters  

# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = GaussianNB()
model.fit(X_train, y_train)

# Evaluar modelo
y_pred = model.predict(X_test)
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))
print("\nMatriz de confusión:")
print(confusion_matrix(y_test, y_pred))