from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import pandas as pd
from sklearn.decomposition import PCA
import seaborn as sns; sns.set(font_scale=1.2)
from sklearn.metrics import accuracy_score

iris= load_iris()
df = pd.read_csv('iris.csv')
features = ['sepal.length', 'sepal.width', 'petal.length', 'petal.width']# Separating out the features
x = df.iloc[:, 0:4].values# Separating out the target
y = df['variety']
x = StandardScaler().fit_transform(x)

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
print(pca.explained_variance_ratio_)

principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

finalDf = pd.concat([principalDf, df[['variety']]], axis = 1)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = ['Setosa', 'Versicolor', 'Virginica']
colors = ['r', 'g', 'b']

for target, color in zip(targets,colors):
    indicesToKeep = finalDf['variety'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)

ax.legend(targets)
ax.grid()
plt.show()

loading_scores = pd.Series(pca.components_[0], index=features)
sorted_loading_socres = loading_scores.abs().sort_values(ascending=False)

top_4_features = sorted_loading_socres[0:4].index.values
print(top_4_features)


def get_features_combinations(nColumns):
    combinations = []
    for i in range(nColumns):
        for j in range(i + 1, nColumns):
            singleCombination = (i, j)
            combinations.append(singleCombination)

    return combinations

def feature_VS_feature_points():
        combinations = get_features_combinations(len(df.iloc[:, 0:4].columns))
        for i in range(len(combinations)):
            feature1_i = combinations[i][0]
            feature2_i = combinations[i][1]

            feature1_name = df.iloc[:, 0:4].columns[feature1_i]
            feature2_name = df.iloc[:, 0:4].columns[feature2_i]

            sns.lmplot(x=feature1_name, y=feature2_name, data=df, hue='variety',
                       legend=True, palette='Set1', fit_reg=False, scatter_kws={"s": 70})

            plt.show()
            i += 1


lr = LogisticRegression(solver='lbfgs', multi_class='auto', tol=0.0001)
df = shuffle(df)
lr.fit(df.iloc[0:100, 0:4], df.iloc[0:100, -1])
y_predict = lr.predict(df.iloc[100:150, 0:4])

y_test = df.iloc[100:150, -1].values

counter_true = 0
counter_false = 0
for i in range(50):
    one = y_predict[i]
    two = y_test[i]
    if one == two:
        counter_true += 1
    else:
        counter_false += 1

print('Score is', (counter_true/len(y_predict)) * 100)

lr2 = LogisticRegression(solver='lbfgs', multi_class='auto', tol=0.02)
df = shuffle(df)
lr2.fit(df.iloc[0:100, [1,2]], df.iloc[0:100, -1])
y_predict = lr2.predict(df.iloc[100:150, [1,2]])

y_test = df.iloc[100:150, -1].values

counter_true = 0
counter_false = 0
for i in range(50):
    one = y_predict[i]
    two = y_test[i]
    if one == two:
        counter_true += 1
    else:
        counter_false += 1

print('Score is', (counter_true/len(y_predict)) * 100)

feature_VS_feature_points()