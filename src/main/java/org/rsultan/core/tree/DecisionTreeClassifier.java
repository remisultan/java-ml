package org.rsultan.core.tree;

import static java.util.Arrays.stream;
import static java.util.Map.Entry.comparingByValue;
import static java.util.Map.entry;
import static java.util.function.Function.identity;
import static java.util.stream.Collectors.counting;
import static java.util.stream.Collectors.groupingBy;
import static java.util.stream.Collectors.toList;
import static java.util.stream.IntStream.range;
import static org.nd4j.linalg.factory.Nd4j.argMax;

import java.util.ArrayList;
import java.util.List;
import java.util.UUID;
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

public class DecisionTreeClassifier
    extends ModelParameters<DecisionTreeClassifier>
    implements Trainable<DecisionTreeClassifier> {

  private static final Logger LOG = LoggerFactory.getLogger(DecisionTreeClassifier.class);
  private final ExecutorService executor = Executors.newCachedThreadPool();
  private final int depth;
  private final ImpurityStrategy strategy;
  private List<String> labelNames;
  private Node tree;
  private ImpurityService impurityService;
  private List<String> featuresNames;

  public DecisionTreeClassifier(int depth, ImpurityStrategy strategy) {
    this.depth = depth;
    this.strategy = strategy;
  }

  @Override
  public DecisionTreeClassifier train(Dataframe dataframe) {
    var dataframeNoResponse = dataframe.mapWithout(responseVariableName);
    var dataframeFeatures = predictorNames.length == 0 ?
        dataframeNoResponse : dataframeNoResponse.select(predictorNames);
    var X = dataframeFeatures.toMatrix();
    featuresNames = stream(dataframeFeatures.getColumns()).map(Column::columnName)
        .collect(toList());
    labelNames = dataframe.get(responseVariableName).stream().sorted().distinct()
        .map(String::valueOf).collect(toList());
    impurityService = strategy.getImpurityService(labelNames.size());
    var columnTemp = UUID.randomUUID().toString();
    var Y = dataframe.map(columnTemp, r -> labelNames.indexOf(r.toString()), responseVariableName)
        .toMatrix(columnTemp);
    this.tree = buildTree(X, Y, depth);
    executor.shutdown();
    return this;
  }

  private Node buildTree(INDArray features, INDArray response, int currentDepth) {
    if (currentDepth < 0) {
      return null;
    }
    LOG.info("Sorting labels per features");
    var classCount = impurityService.getClassCount(response);
    LOG.info("Class count {}", classCount);
    int predictedLabel = argMax(classCount).getInt(0, 0);
    LOG.info("Predicted feature node {} for depth {}", labelNames.get(predictedLabel),
        depth - currentDepth);
    LOG.info("Creating matrix feature");
    var sortedLabels = getSortedLabelsPerFeature(features, response);
    LOG.info("Computing best split");
    var gain = getBestSplit(classCount, sortedLabels.get(0), sortedLabels.get(1));
    var leftNode = buildLeftNode(features, response, currentDepth, gain);
    var rightNode = buildRightNode(features, response, currentDepth, gain);
    return new Node(gain.feature(), gain.threshold(), predictedLabel, getFuture(leftNode),
        getFuture(rightNode));
  }

  private List<INDArray> getSortedLabelsPerFeature(INDArray features, INDArray response) {
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

  private BestSplit getBestSplit(INDArray classCount, INDArray sortedFeatures,
      INDArray sortedLabels) {
    final double maxRows = sortedLabels.rows();
    var bestSplit = new BestSplit(0, 0, impurityService.compute(classCount).getDouble(0, 0));
    for (int featureIdx = 0; featureIdx < sortedFeatures.columns(); featureIdx++) {
      var thresholds = sortedFeatures.getColumn(featureIdx);
      var labels = sortedLabels.getColumn(featureIdx);
      var left = Nd4j.zeros(1, labelNames.size());
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

  private Future<Node> buildLeftNode(INDArray features, INDArray response, int currentDepth,
      BestSplit split) {
    var feature = features.getColumn(split.feature());
    var indices = range(0, feature.columns()).parallel()
        .filter(idx -> feature.getDouble(idx) < split.threshold())
        .toArray();
    return submitFuture(() -> buildNode(features, response, currentDepth, indices));
  }

  private Future<Node> buildRightNode(INDArray features, INDArray response, int currentDepth,
      BestSplit split) {
    var feature = features.getColumn(split.feature());
    var indices = range(0, feature.columns()).parallel()
        .filter(idx -> feature.getDouble(idx) >= split.threshold())
        .toArray();
    return submitFuture(() -> buildNode(features, response, currentDepth, indices));
  }

  private Node buildNode(INDArray features, INDArray response, int currentDepth, int[] indices) {
    return indices.length == 0 ? null
        : buildTree(features.getRows(indices), response.getRows(indices), currentDepth - 1);
  }

  @Override
  public Dataframe predict(Dataframe dataframe) {
    var predictions = new Column<String>(predictionColumnName, new ArrayList<>());
    range(0, dataframe.getRows()).forEach(row -> {
      var node = tree;
      while (node.left() != null) {
        var featureName = featuresNames.get(node.feature());
        double featureValue = dataframe.<Number>get(featureName).get(row).doubleValue();
        if (featureValue < node.featureThreshold()) {
          node = node.left();
        } else {
          node = node.right();
        }
      }
      var stringLabel = labelNames.get(node.predictedLabel());
      predictions.values().add(row, stringLabel);
    });
    return dataframe.addColumn(predictions);
  }

  private <T> Future<T> submitFuture(Callable<T> supplier) {
    return executor.submit(supplier);
  }

  private <T> T getFuture(Future<T> future) {
    try {
      return future.get();
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
  }
}
