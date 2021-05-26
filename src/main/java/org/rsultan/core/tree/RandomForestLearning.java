package org.rsultan.core.tree;

import static java.lang.Math.min;
import static java.util.Arrays.stream;
import static java.util.stream.Collectors.toList;
import static java.util.stream.IntStream.range;

import java.util.Collections;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.rsultan.core.ModelParameters;
import org.rsultan.core.Trainable;
import org.rsultan.dataframe.Column;
import org.rsultan.dataframe.Dataframe;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public abstract class RandomForestLearning extends ModelParameters<RandomForestLearning>
    implements Trainable<RandomForestLearning> {

  private static final Logger LOG = LoggerFactory.getLogger(RandomForestLearning.class);
  private final int numberOfEstimators;
  private final ExecutorService executor;

  protected double sampleSizeRatio = 0.25;
  protected int treeDepth = 1;
  protected int sampleFeatures = 0;

  protected List<?> responses;
  protected List<?> featureNames;

  private List<? extends DecisionTreeLearning> trees;

  public RandomForestLearning(int numberOfEstimators) {
    this.numberOfEstimators = numberOfEstimators < 1 ? 10 : numberOfEstimators;
    executor = Executors.newFixedThreadPool(this.numberOfEstimators);
  }

  protected abstract DecisionTreeLearning buildDecisionTreeLearning();

  protected abstract INDArray buildY(Dataframe dataframe);

  protected abstract List<?> getResponseValues(Dataframe dataframe);

  protected abstract int getFeatureSampleSize(int numberOfFeatures);

  protected abstract List<?> getFinalPredictions(INDArray predictionMatrix);

  @Override
  public RandomForestLearning train(Dataframe dataframe) {
    var dfNoResponse = dataframe.mapWithout(responseVariableName);
    var dfFeatures =
        predictorNames.length == 0 ? dfNoResponse : dfNoResponse.select(predictorNames);

    var columns = List.of(dfFeatures.getColumns());
    featureNames = columns.stream().map(Column::columnName).collect(toList());
    var featureIndices = featureNames.stream().map(featureNames::indexOf).collect(toList());
    responses = getResponseValues(dataframe);

    var X = dfFeatures.toMatrix();
    var Y = buildY(dataframe);

    int rowSampleSize = (int) (X.rows() * sampleSizeRatio);
    int featureSampleSize = getFeatureSampleSize(X.columns());

    trees = range(0, numberOfEstimators)
        .peek(idx -> LOG.debug("Tree number: " + (idx + 1)))
        .mapToObj(idx -> buildDecisionTreeLearning())
        .map(decisionTreeLearning -> {
          var subRowIndices = getSampleIndices(X.rows(), min(rowSampleSize, X.rows()));
          var subFeatureIndices = getSampleIndices(X.columns(),
              min(featureSampleSize, X.columns()));
          var Xsampled = X.getRows(subRowIndices).getColumns(subFeatureIndices);
          var Ysampled = Y.getRows(subRowIndices);
          var localFeatures = stream(subFeatureIndices).mapToObj(featureIndices::get)
              .collect(toList());
          return executor.submit(
              () -> decisionTreeLearning.train(Xsampled, Ysampled, localFeatures, responses)
          );
        }).map(this::getFuture).collect(toList());
    executor.shutdown();
    return this;
  }

  private DecisionTreeLearning getFuture(Future<? extends DecisionTreeLearning> future) {
    try {
      return future.get();
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
  }

  private int[] getSampleIndices(int m, int rowSampleSize) {
    List<Integer> collect = range(0, m).boxed().collect(toList());
    Collections.shuffle(collect);
    return collect.subList(0, rowSampleSize).stream().mapToInt(i -> i).toArray();
  }

  @Override
  public Dataframe predict(Dataframe dataframe) {
    var allPredictions = this.trees.parallelStream()
        .map(tree -> tree.<Double>rawPredict(dataframe))
        .map(Nd4j::create)
        .collect(toList());
    long rows = allPredictions.size();
    long cols = allPredictions.get(0).columns();
    var predictionMatrix = Nd4j.create(allPredictions, rows, cols);
    List<?> predictions = getFinalPredictions(predictionMatrix);
    return dataframe.addColumn(new Column<>(predictionColumnName, predictions));
  }

  public RandomForestLearning setSampleSizeRatio(double sampleSizeRatio) {
    this.sampleSizeRatio = sampleSizeRatio;
    return this;
  }

  public RandomForestLearning setTreeDepth(int treeDepth) {
    this.treeDepth = treeDepth;
    return this;
  }

  public RandomForestLearning setSampleFeatureSize(int sampleFeatures) {
    this.sampleFeatures = sampleFeatures;
    return this;
  }
}
