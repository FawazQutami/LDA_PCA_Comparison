import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA


if __name__ == '__main__':
    wine = datasets.load_wine()
    X = wine.data
    y = wine.target
    labels = wine.target_names

    lda = LinearDiscriminantAnalysis(n_components=2).fit(X, y).transform(X)
    pca = PCA(n_components=2).fit_transform(X)

    with plt.style.context('ggplot'):
        fig, axes = plt.subplots(1, 2, figsize=[12, 4])
        colors = ['navy', 'turquoise', 'darkorange']

        for color, i, lbl in zip(colors, [0, 1, 2], labels):
            axes[0].scatter(lda[y == i, 0]
                            , lda[y == i, 1]
                            , alpha=.8
                            , label=lbl
                            , color=color)
            axes[1].scatter(pca[y == i, 0]
                            , pca[y == i, 1]
                            , alpha=.8
                            , label=lbl
                            , color=color)

        axes[0].title.set_text('LDA')
        axes[1].title.set_text('PCA')
        axes[0].set_xlabel('LD 1')
        axes[0].set_ylabel('LD 2')
        axes[1].set_xlabel('PC 1')
        axes[1].set_ylabel('PC 2')

        plt.show()
