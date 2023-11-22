import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
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

print("Salvando o dataframe antes da classificação.")
df.to_csv('D:\\PCA GARCIA\\antes_classificacao.csv', index=False)

print("Iniciando a operação de aplicação do SVD e PCA. Esta operação irá transformar os dados originais em componentes principais.")
X = df.drop('label', axis=1)
y = df['label']
X = StandardScaler().fit_transform(X)

svd = TruncatedSVD(n_components=3)
principalComponents_svd = svd.fit_transform(X)

pca = PCA(n_components=3)
principalComponents_pca = pca.fit_transform(X)

print("Calculando os autovalores da matriz.")
eigenvalues = np.linalg.eigvals(X.T @ X)
print('Autovalores: ', eigenvalues)

print("Salvando o dataframe após a classificação.")
df_svd = pd.DataFrame(data=principalComponents_svd, columns=['pc1', 'pc2', 'pc3'])
df_svd = df_svd.reset_index(drop=True)
df_svd['label'] = y.values
df_svd.to_csv('D:\\PCA GARCIA\\depois_classificacao.csv', index=False)

print("Calculando a acurácia da classificação.")
accuracy = accuracy_score(y, df_svd['label'])
print('Acurácia da classificação: ', accuracy)

print("Plotando os gráficos.")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), subplot_kw={'projection': '3d'})

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

ax2.set_xlabel('Principal Component 1', fontsize=15)
ax2.set_ylabel('Principal Component 2', fontsize=15)
ax2.set_zlabel('Principal Component 3', fontsize=15)
ax2.set_title('3 component PCA - Após a classificação', fontsize=20)

for target, color in zip(targets, colors):
    indicesToKeep = df_svd['label'] == target
    ax2.scatter(df_svd.loc[indicesToKeep, 'pc1'],
                df_svd.loc[indicesToKeep, 'pc2'],
                df_svd.loc[indicesToKeep, 'pc3'],
                c=color,
                s=50)
ax2.legend(targets)
ax2.grid()

plt.show()

print("--- %s seconds ---" % (time.time() - start_time))
