# Comparing PCA, t-SNE and UMAP for Regression on the Diabetes Dataset

**Jibril Hussaini**
hussainijibril0@gmail.com
Department of Computer Science, Bishop's University

## Abstract

This project is a comparative study of three dimensionality reduction methods, namely Principal
Component Analysis (PCA), t-Distributed Stochastic Neighbor Embedding (t-SNE) and Uniform Manifold
Approximation and Projection (UMAP), and how each one affects a linear regression model that
predicts diabetes disease progression. I reduce the ten baseline variables of the diabetes dataset
to one, two and three dimensions with each method, fit linear regression on the resulting
coordinates, and compare the mean squared error. I also use Lasso regression on the full feature
set as a reference. The main result is that PCA keeps the linear relationship with the target and
gives a low error even in two dimensions, while t-SNE and UMAP organise the points into neat groups
but lose much of the linear signal, so they need a third dimension before linear regression
catches up.

## 1. Introduction

Medical datasets often carry many measurements per patient, and looking at all of them at once is
hard. The diabetes dataset used here records ten baseline values for 442 patients together with a
score that measures how far the disease has progressed after one year. Ten variables is already too
many to plot directly, so we need a way to bring the data down to two or three dimensions that we
can actually look at on a screen. Dimensionality reduction does this, but every method makes its own
trade-off: some keep the overall geometry of the data, others keep small neighbourhoods together,
and the choice changes what we can do with the result afterwards. Because the diabetes target is a
continuous number, a natural way to judge a reduction is to ask how well a simple regression model
can still predict the target from the reduced coordinates. That question ties visualization and
prediction together and is the focus of this report.

**Objectives.** Reduce the diabetes data with PCA, t-SNE and UMAP; study how each method's main
parameter changes the picture; and measure how the reduction affects the error of a linear
regression that predicts disease progression.

## 2. Background

**Data visualization** is the use of charts and plots to make patterns in data visible. With a few
variables we can use scatter plots, bar charts and heatmaps directly, but high-dimensional data
first has to be projected down to two or three dimensions before we can see it. The way that
projection is done decides which patterns survive and which ones are hidden.

**Regression** fits a function that predicts a continuous output from a set of inputs. Linear
regression assumes a straight-line (or flat-plane) relationship and chooses the coefficients that
minimise the squared error between the predictions and the true values. Mean squared error (MSE) is
the average of those squared differences, so a smaller MSE means a better fit. Lasso regression adds
a penalty on the size of the coefficients, controlled by a parameter `alpha`: as `alpha` grows the
coefficients are pushed toward zero, which removes weak variables and simplifies the model but, if
pushed too far, throws away useful signal as well.

**Dimensionality reduction** replaces many variables with a few new ones that still describe most of
the data. The three methods studied here work in very different ways:

- **PCA** is linear. It finds the directions of greatest variance in the data and uses them as new
  axes called principal components. These axes are ordered by how much variance they explain and are
  uncorrelated with each other. PCA keeps global structure and is easy to interpret through its
  loadings, which show how each original variable contributes to a component, but it can only
  capture straight-line structure.

- **t-SNE** is non-linear and focuses on local neighbourhoods. It tries to place points that are
  close in the original space close together in the map, which makes clusters stand out clearly. The
  cost is that the axes carry no meaning, distances between separate clusters are not reliable, and
  the result depends on the `perplexity` parameter, which sets roughly how many neighbours each
  point pays attention to.

- **UMAP** is also non-linear and builds a graph of nearest neighbours, then lays that graph out in
  low dimensions. It is usually faster than t-SNE and keeps more of the global arrangement of the
  groups. Its main parameter, `n_neighbors`, trades local detail (small values) against global
  shape (large values).

## 3. Experiments and evaluation

### 3.1 Dataset

The dataset has 442 patients and eleven columns: age, sex, body-mass index (BMI), average blood
pressure (BP), six blood-serum measurements (S1 to S6), and the target `Y`. There are no missing
values. All variables were scaled to the range [0, 1] with `MinMaxScaler` for the visual analysis.
For the regression experiments the inputs were scaled the same way while the target was kept on its
original scale so that the regularisation sweep stayed meaningful.

A correlation check ranks the inputs by how strongly they move with the target. BMI and S5 are the
two strongest, with correlations of about +0.59 and +0.57, followed by BP (+0.44), S4 (+0.43), S3
(−0.40) and S6 (+0.38); sex is almost unrelated to the target. The scatter of BMI against S5,
coloured by the target, shows the progression score rising toward the top-right corner, which
confirms that these two variables already carry a large part of the signal.

### 3.2 Results for each method

**Lasso (reference on all ten variables).** The smallest error, MSE = 2859.7, came at `alpha` = 0,
which is ordinary linear regression with no penalty. At `alpha` = 1 the error rose slightly to
3000.4, and from `alpha` = 10 onward it sat flat at 5929.9 (the variance of the target) because
the penalty had shrunk every coefficient to zero and the model was just predicting the mean. With
only ten variables, all of which carry some signal, regularisation only hurts here; the value of the
experiment is in seeing the coefficients collapse to zero as `alpha` grows.

**PCA.** The first component explains 49.8% of the variance and the second 19.7%, so two components
already cover about 69% of the data. The loadings show BMI, BP, S5 and the serum measures pulling
together along the first component, which behaves like an overall severity axis. Linear regression
on PC1 alone gave MSE = 5748.9; adding PC2 dropped it sharply to 3833.5, and a third component
barely changed it (3804.5). The strong improvement from one to two components shows that the
linear relationship with the target lives mainly in the first two principal directions.

**t-SNE.** With perplexity 5 the map breaks into many small islands; as perplexity grows to 50 the
points merge into two broad bands, so the parameter clearly controls how local or global the picture
is. Regression told a sharper story: on the first dimension the error was 5916.9 and on the first
two it was essentially the same (5916.9), both equal to just predicting the mean. Because t-SNE axes
are arbitrary, a straight line through them carries no information about the target. Only when a
third t-SNE dimension was added did the error fall to 3450.1, the best three-dimensional result of
any method.

**UMAP.** Changing `n_neighbors` from 5 to 50 moved the layout from many fine clusters toward a
smoother global shape, the same trade-off seen with t-SNE perplexity. Regression on the first UMAP
dimension gave 5811.2, the first two gave 4696.4, and three gave 4684.9. UMAP therefore kept more
linear signal than t-SNE in two dimensions but did not match PCA.

### 3.3 Comparison of the best results

| Method | MSE (1 dim) | MSE (2 dims) | MSE (3 dims) | R² (3 dims) |
|--------|------------:|-------------:|-------------:|------------:|
| PCA    | 5748.9 | 3833.5 | 3804.5 | 0.358 |
| t-SNE  | 5916.9 | 5916.9 | 3450.1 | 0.418 |
| UMAP   | 5811.2 | 4696.4 | 4684.9 | 0.210 |

In two dimensions PCA is the clear winner (3833.5 against 4696.4 for UMAP and 5916.9 for t-SNE). In
three dimensions t-SNE gives the lowest error and the highest R² (0.418), with PCA close behind and
UMAP last. For reference, predicting the mean gives an MSE of about 5929.9, so t-SNE and UMAP in low
dimensions sit right at that baseline.

### 3.4 Discussion

The results line up with the theory. PCA is linear, so the new axes stay linearly related to the
target and ordinary linear regression works on them straight away. Two principal components were
enough to cut the error by a third. t-SNE and UMAP are built to separate neighbourhoods, not to
preserve straight-line relationships, so their first two axes look tidy on a plot but are almost
useless for linear prediction; t-SNE only became competitive once a third dimension gave the linear
model more room to work with. There is a clear gap between a method that *looks* good and one that
is *useful for prediction*: the cleanest cluster plots (t-SNE, UMAP) were the worst for
two-dimensional regression. The parameter studies reinforced this. Perplexity and `n_neighbors`
both shift the balance between local detail and global shape, but neither turns a non-linear
embedding into a linearly predictable one.

## 4. Conclusion

This project compared PCA, t-SNE and UMAP as front-ends to linear regression on the diabetes
dataset. PCA was the most reliable choice: it is fast, its components are interpretable through their
loadings, and it kept the linear relationship with the target so well that two components already
gave a low error. t-SNE and UMAP produced clearer-looking groupings but stripped out the linear
signal, and only a third dimension let t-SNE recover and post the best three-dimensional score.
The lesson is that the best reduction depends on the goal. For visual exploration of clusters, t-SNE
and UMAP are strong; for keeping a target predictable by a linear model, PCA is the safer and more
interpretable option.
