package org.rsultan.core.tree;

import static java.util.Arrays.stream;
import static java.util.Objects.nonNull;
import static java.util.stream.Collectors.toList;
import static java.util.stream.IntStream.range;

import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.rsultan.core.ModelParameters;
import org.rsultan.core.Trainable;
import org.rsultan.core.tree.domain.BestSplit;
import org.rsultan.core.tree.domain.Node;
import org.rsultan.core.tree.impurity.ImpurityService;
import org.rsultan.core.tree.impurity.ImpurityStrategy;
import org.rsultan.dataframe.Column;
import org.rsultan.dataframe.Dataframe;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public abstract class DecisionTreeLearning extends ModelParameters<DecisionTreeLearning> implements
    Trainable<DecisionTreeLearning> {

  private static final Logger LOG = LoggerFactory.getLogger(DecisionTreeLearning.class);

  protected transient final ExecutorService executor = Executors.newCachedThreadPool();
  protected final int depth;
  protected final ImpurityStrategy strategy;

  protected Node tree;
  protected List<?> responses;
  protected List<?> features;
  protected ImpurityService impurityService;

  public DecisionTreeLearning(int depth, ImpurityStrategy strategy) {
    this.depth = depth > 0 ? depth : 1;
    this.strategy = strategy;
  }

  protected abstract <T extends Number> T computePredictedResponse(INDArray array);

  protected abstract <T> T getNodePrediction(Node number);

  protected abstract List<?> getResponseValues(Dataframe dataframe);

  protected abstract INDArray buildY(Dataframe dataframe);

  @Override
  public DecisionTreeLearning train(Dataframe dataframe) {
    var dfNoResponse = dataframe.mapWithout(responseVariableName);
    var dfFeatures =
        predictorNames.length == 0 ? dfNoResponse : dfNoResponse.select(predictorNames);
    var featureNames = stream(dfFeatures.getColumns()).map(Column::columnName).collect(toList());
    var responses = getResponseValues(dataframe);
    return train(dfFeatures.toMatrix(), buildY(dataframe), featureNames, responses);
  }

  DecisionTreeLearning train(INDArray X, INDArray Y, List<?> features, List<?> responses) {
    impurityService = strategy.getImpurityService(responses.size());
    this.features = features;
    this.responses = responses;
    this.tree = buildTree(X, Y, depth);
    return this;
  }

  @Override
  public Dataframe predict(Dataframe dataframe) {
    return dataframe.addColumn(new Column<>(predictionColumnName, rawPredict(dataframe)));
  }

  protected <T> List<T> rawPredict(Dataframe dataframe) {
    return range(0, dataframe.getRowSize()).mapToObj(row -> {
      var node = tree;
      while (nonNull(node.left())) {
        var featureName = getPredictionNodeFeatureName(node);
        double featureValue = dataframe.<Number>get(featureName).get(row).doubleValue();
        node = featureValue < node.featureThreshold() ? node.left() : node.right();
      }
      return this.<T>getNodePrediction(node);
    }).collect(toList());
  }

  protected abstract <T> T getPredictionNodeFeatureName(Node node);

  protected Node buildTree(INDArray features, INDArray response, int currentDepth) {
    if (currentDepth < 0) {
      return null;
    }
    LOG.debug("Depth: " + (depth - currentDepth + 1));
    LOG.debug("Sorting labels per features");
    var sortedLabels = getSortedLabelsPerFeature(features, response);
    LOG.debug("Computing best split");
    var gain = getBestSplit(response, sortedLabels.get(0), sortedLabels.get(1));
    LOG.debug("Computing children nodes");
    var leftNode = buildLeftNode(features, response, currentDepth, gain);
    var rightNode = buildRightNode(features, response, currentDepth, gain);
    var node = new Node(
        gain.feature(),
        gain.threshold(),
        computePredictedResponse(response),
        getFuture(leftNode),
        getFuture(rightNode));
    if (currentDepth == depth) {
      executor.shutdown();
    }
    return node;
  }

  protected List<INDArray> getSortedLabelsPerFeature(INDArray features, INDArray response) {
    var sortedLabelsAndFeatures = range(0, features.columns()).parallel()
        .mapToObj(col -> features.getColumn(col, false))
        .map(col -> Nd4j.create(col.toDoubleVector(), features.rows(), 1))
        .map(column -> Nd4j.concat(1, column, response))
        .map(matrix -> Nd4j.sortRows(matrix, 0, true))
        .collect(toList());
    return List.of(
        Nd4j.create(
            sortedLabelsAndFeatures.stream().map(mat -> mat.getColumn(0)).collect(toList()),
            features.columns(), features.rows()
        ).transpose(),
        Nd4j.create(
            sortedLabelsAndFeatures.stream().map(mat -> mat.getColumn(1)).collect(toList()),
            features.columns(), features.rows()
        ).transpose()
    );
  }

  protected BestSplit getBestSplit(INDArray response, INDArray sortedFeatures,
      INDArray sortedLabels) {
    var classCount = impurityService.getClassCount(response);
    final double maxRows = sortedLabels.rows();
    var bestSplit = new BestSplit(0, 0, impurityService.compute(classCount).getDouble(0, 0));
    for (int featureIdx = 0; featureIdx < sortedFeatures.columns(); featureIdx++) {
      var thresholds = sortedFeatures.getColumn(featureIdx);
      var labels = sortedLabels.getColumn(featureIdx);
      var left = Nd4j.zeros(1, responses.size());
      var right = classCount.dup();
      for (int splitIdx = 1; splitIdx < maxRows; splitIdx++) {
        var classVal = labels.getInt(splitIdx - 1);
        left.putScalar(classVal, left.getInt(classVal) + 1);
        right.putScalar(classVal, right.getInt(classVal) - 1);
        double currentThreshold = thresholds.getDouble(splitIdx);
        if (currentThreshold != thresholds.getDouble(splitIdx - 1)) {
          var impurityLeft = impurityService.compute(left).mul(splitIdx / maxRows);
          var impurityRight = impurityService.compute(right).mul((maxRows - splitIdx) / maxRows);
          var impurity = impurityLeft.add(impurityRight).getDouble(0, 0);
          if (impurity < bestSplit.impurity()) {
            bestSplit = new BestSplit(featureIdx, currentThreshold, impurity);
          }
        }
      }
    }
    return bestSplit;
  }

  protected Future<Node> buildLeftNode(INDArray features, INDArray response, int currentDepth,
      BestSplit split) {
    var feature = features.getColumn(split.feature());
    var indices = range(0, feature.columns()).parallel()
        .filter(idx -> feature.getDouble(idx) < split.threshold())
        .toArray();
    return submitFuture(() -> buildNode(features, response, currentDepth, indices));
  }

  protected Future<Node> buildRightNode(INDArray features, INDArray response, int currentDepth,
      BestSplit split) {
    var feature = features.getColumn(split.feature());
    var indices = range(0, feature.columns()).parallel()
        .filter(idx -> feature.getDouble(idx) >= split.threshold())
        .toArray();
    return submitFuture(() -> buildNode(features, response, currentDepth, indices));
  }

  protected Node buildNode(INDArray features, INDArray response, int currentDepth, int[] indices) {
    return indices.length == 0 ? null
        : buildTree(features.getRows(indices), response.getRows(indices), currentDepth - 1);
  }

  protected <T> Future<T> submitFuture(Callable<T> supplier) {
    return executor.submit(supplier);
  }

  protected <T> T getFuture(Future<T> future) {
    try {
      return future.get();
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
  }
}
