import pandas as pd
import numpy as np
import copy
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import dash
from dash.dependencies import Input, Output, State
import warnings
# User defined modules
from beginner_projects.iris_dataset.app_layout import app_layout, plot_iris

warnings.filterwarnings("ignore", category=DeprecationWarning)

available_models = {'logistic regression': LogisticRegression(solver='lbfgs', multi_class='ovr'),
                    'k neighbors classifier': KNeighborsClassifier(),
                    'decision tree classifier': DecisionTreeClassifier()}


def plot_output(df, model, x='sepal_length', y='sepal_width'):
    species = df.species.unique()
    plt.scatter(df[df.species == species[0]][x], df[df.species == species[0]][y], marker='+')
    plt.scatter(df[df.species == species[1]][x], df[df.species == species[1]][y],
                c='green', marker='o')

    h = .02  # step size in the mesh
    # create a mesh to plot in
    x1_samples = [df[df.species == species[0]][x], df[df.species == species[0]][y]]
    x2_samples = [df[df.species == species[1]][x], df[df.species == species[1]][y]]
    X = np.concatenate((x1_samples, x2_samples), axis = 0)
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel(), xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, cmap=plt.cm.Paired)
    plt.xlabel(x, fontsize=15)
    plt.ylabel(y, fontsize=15)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.show()


def classify(model_name, ratio=0.2):
    data_frame = copy.deepcopy(iris)
    le = LabelEncoder()
    data_frame['species'] = le.fit_transform(data_frame.species)
    X = data_frame.drop(columns=['species'])
    y = data_frame['species']
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=ratio, random_state=10)

    model = available_models[model_name]
    model.fit(x_train, y_train)
    print('accuracy: ', model.score(x_test, y_test))

    plot_output(data_frame, model)


iris = pd.read_csv("iris.csv")
app = dash.Dash(__name__)
server = app.server
app.layout = app_layout(iris, 'sepal_length', 'sepal_width', available_models)


@app.callback(Output(component_id='iris-plot', component_property='figure'),
              [Input(component_id='x-dropdown', component_property='value'),
               Input(component_id='y-dropdown', component_property='value'),
               Input(component_id='classifier', component_property='n_clicks')
               ],
              [State(component_id='model-selector', component_property='value')]
              )
def update_plot(x, y, _, model_name):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    new_fig = plot_iris(iris, x, y)
    if 'classifier' in changed_id:
        classify(model_name)
    return new_fig


if __name__ == '__main__':
    app.run_server()
