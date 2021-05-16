package org.rsultan.core.tree;

import static java.util.Arrays.stream;
import static java.util.Objects.nonNull;
import static java.util.stream.Collectors.toList;
import static java.util.stream.IntStream.range;

import java.util.ArrayList;
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

public abstract class DecisionTreeLearning
    extends ModelParameters<DecisionTreeLearning>
    implements Trainable<DecisionTreeLearning> {

  private static final Logger LOG = LoggerFactory.getLogger(DecisionTreeLearning.class);
  protected final ExecutorService executor = Executors.newCachedThreadPool();
  protected final int depth;
  protected final ImpurityStrategy strategy;

  protected Node tree;
  protected List<?> responseValues;
  protected List<String> featuresNames;
  protected ImpurityService impurityService;

  public DecisionTreeLearning(int depth, ImpurityStrategy strategy) {
    this.depth = depth > 0 ? depth : 1;
    this.strategy = strategy;
  }

  protected abstract <T extends Number> T computePredictedResponse(INDArray array);

  protected abstract Object getNodePrediction(Node node);

  protected abstract List<?> getResponseValues(Dataframe dataframe);

  protected abstract INDArray buildY(Dataframe dataframe);

  @Override
  public DecisionTreeLearning train(Dataframe dataframe) {
    var dataframeNoResponse = dataframe.mapWithout(responseVariableName);
    var dataframeFeatures = predictorNames.length == 0 ?
        dataframeNoResponse : dataframeNoResponse.select(predictorNames);

    featuresNames = stream(dataframeFeatures.getColumns()).map(Column::columnName)
        .collect(toList());
    responseValues = getResponseValues(dataframe);
    impurityService = strategy.getImpurityService(responseValues.size());

    var X = dataframeFeatures.toMatrix();
    var Y = this.buildY(dataframe);
    this.tree = buildTree(X, Y, depth);
    return this;
  }

  @Override
  public Dataframe predict(Dataframe dataframe) {
    var predictions = new Column<>(predictionColumnName, new ArrayList<>());
    range(0, dataframe.getRows()).mapToObj(row -> {
      var node = tree;
      while (nonNull(node.left())) {
        var featureName = featuresNames.get(node.feature());
        double featureValue = dataframe.<Number>get(featureName).get(row).doubleValue();
        node = featureValue < node.featureThreshold() ? node.left() : node.right();
      }
      return getNodePrediction(node);
    }).forEach(predictions.values()::add);
    return dataframe.addColumn(predictions);
  }

  protected Node buildTree(INDArray features, INDArray response, int currentDepth) {
    if (currentDepth < 0) {
      return null;
    }
    LOG.info("Sorting labels per features");
    var sortedLabels = getSortedLabelsPerFeature(features, response);
    LOG.info("Computing best split");
    var gain = getBestSplit(response, sortedLabels.get(0), sortedLabels.get(1));
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
      var left = Nd4j.zeros(1, responseValues.size());
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
