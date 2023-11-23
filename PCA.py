import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from PIL import Image
import time

start_time = time.time()

print("Iniciando a operação de carregamento de imagens. Esta operação irá carregar todas as imagens dos diretórios especificados.")
def load_images(folder):
    images = []
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images

dogs = load_images('D:\\PCA GARCIA\\dog')
cats = load_images('D:\\PCA GARCIA\\cat')

print("Iniciando a operação de pré-processamento. Esta operação irá redimensionar as imagens, convertê-las para tons de cinza e achatar as imagens.")
def process_images(image_list):
    processed = []
    for img in image_list:
        img = img.resize((128, 128))
        img = img.convert('L')
        img = np.array(img).flatten()
        processed.append(img)
    return processed

dogs = process_images(dogs)
cats = process_images(cats)

print("Iniciando a operação de criação do dataframe. Esta operação irá criar um dataframe pandas a partir das imagens processadas.")
df_dogs = pd.DataFrame(dogs)
df_dogs['label'] = 'dog'
df_cats = pd.DataFrame(cats)
df_cats['label'] = 'cat'
df = pd.concat([df_dogs, df_cats])

print("O tamanho da matriz antes da classificação é: ", df.shape)

print("Dividindo os dados em conjuntos de treinamento e teste.")
X = df.drop('label', axis=1)
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Normalizando os dados.")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Calculando os autovalores e autovetores da matriz.")
eigenvalues, eigenvectors = np.linalg.eig(X_train.T @ X_train)
print('Autovalores: ', eigenvalues)
print('Autovetores: ', eigenvectors)

print("Aplicando PCA aos dados.")
pca = PCA(n_components=3)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

print("O tamanho da matriz após a classificação é: ", X_train_pca.shape)

print("Treinando o classificador SVC nos dados de treinamento.")
svc = SVC()
svc.fit(X_train_pca, y_train)

print("Fazendo previsões nos dados de teste.")
y_pred = svc.predict(X_test_pca)

print("Calculando a acurácia da classificação.")
accuracy = accuracy_score(y_test, y_pred)
print('Acurácia da classificação: ', accuracy)

print("Plotando os gráficos.")
# Plotar gráficos
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), subplot_kw={'projection': '3d'})

# Antes da classificação
ax1.set_xlabel('Feature 1', fontsize=15)
ax1.set_ylabel('Feature 2', fontsize=15)
ax1.set_zlabel('Feature 3', fontsize=15)
ax1.set_title('Dados originais - Antes da classificação', fontsize=20)

targets = ['dog', 'cat']
colors = ['b', 'r']
for target, color in zip(targets, colors):
    indicesToKeep = df['label'] == target
    ax1.scatter(df.loc[indicesToKeep, df.columns[0]],
                df.loc[indicesToKeep, df.columns[1]],
                df.loc[indicesToKeep, df.columns[2]],
                c=color,
                s=50)
ax1.legend(targets)
ax1.grid()

# Após a classificação
ax2.set_xlabel('Principal Component 1', fontsize=15)
ax2.set_ylabel('Principal Component 2', fontsize=15)
ax2.set_zlabel('Principal Component 3', fontsize=15)
ax2.set_title('3 component PCA - Após a classificação', fontsize=20)

df_pca = pd.DataFrame(data=X_train_pca, columns=['pc1', 'pc2', 'pc3'])
df_pca['label'] = y_train.values

for target, color in zip(targets, colors):
    indicesToKeep = df_pca['label'] == target
    ax2.scatter(df_pca.loc[indicesToKeep, 'pc1'],
                df_pca.loc[indicesToKeep, 'pc2'],
                df_pca.loc[indicesToKeep, 'pc3'],
                c=color,
                s=50)
ax2.legend(targets)
ax2.grid()

plt.show()

print("--- %s seconds ---" % (time.time() - start_time))