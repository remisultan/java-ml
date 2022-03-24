package org.rsultan.core.ensemble.rf;

import static java.lang.Math.min;
import static java.util.Arrays.stream;
import static java.util.Objects.isNull;
import static java.util.stream.Collectors.toList;
import static java.util.stream.IntStream.range;
import static org.apache.commons.lang3.RandomUtils.nextLong;

import java.util.Collections;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.stream.LongStream;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.rsultan.core.ModelParameters;
import org.rsultan.core.Trainable;
import org.rsultan.core.tree.DecisionTreeLearning;
import org.rsultan.dataframe.Column;
import org.rsultan.dataframe.Dataframe;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public abstract class RandomForestLearning extends ModelParameters<RandomForestLearning>
    implements Trainable<RandomForestLearning> {

  private static final Logger LOG = LoggerFactory.getLogger(RandomForestLearning.class);
  private final int numberOfEstimators;

  protected double sampleSizeRatio = 0.25;
  protected int treeDepth = 1;
  protected int sampleFeatures = 0;

  protected List<?> responses;
  protected List<? extends DecisionTreeLearning> trees;

  public RandomForestLearning(int numberOfEstimators) {
    this.numberOfEstimators = numberOfEstimators < 1 ? 10 : numberOfEstimators;
  }

  protected abstract DecisionTreeLearning buildDecisionTreeLearning();

  protected abstract INDArray buildY(Dataframe dataframe);

  protected abstract List<?> getResponseValues(Dataframe dataframe);

  protected abstract int getFeatureSampleSize(int numberOfFeatures);

  protected abstract List<?> getFinalPredictions(INDArray predictionMatrix);

  protected abstract List<INDArray> getTreePredictions(INDArray predictionMatrix);

  @Override
  public RandomForestLearning train(Dataframe dataframe) {
    var dfNoResponse = dataframe.copy().mapWithout(responseVariableName);
    var dfFeatures =
        predictorNames.length == 0 ? dfNoResponse.copy()
            : dfNoResponse.copy().select(predictorNames);

    responses = getResponseValues(dataframe);

    var X = dfFeatures.toMatrix();
    var Y = buildY(dataframe);

    shuffle(X, Y);

    int rowSampleSize = X.rows() < 10 ? X.rows() : (int) (X.rows() * sampleSizeRatio);
    int featureSampleSize = getFeatureSampleSize(X.columns());
    var subFeatureIndices = getSampleIndices(X.columns(), min(featureSampleSize, X.columns()));

    trees = range(0, numberOfEstimators)
        .peek(i -> LOG.info("Tree number: {}", i))
        .mapToObj(i -> LongStream.range(0, rowSampleSize)
            .map(idx -> nextLong(0, X.rows()))
            .toArray())
        .map(NDArrayIndex::indices)
        .map(sampleIndices -> {
          var Xsampled = X.get(sampleIndices).getColumns(subFeatureIndices);
          var Ysampled = Y.get(sampleIndices);
          return buildDecisionTreeLearning().train(Xsampled, Ysampled);
        }).collect(toList());
    return this;
  }

  private int[] getSampleIndices(int m, int rowSampleSize) {
    List<Integer> collect = range(0, m).boxed().collect(toList());
    Collections.shuffle(collect);
    return collect.subList(0, rowSampleSize).stream().mapToInt(i -> i).toArray();
  }

  @Override
  public Dataframe predict(Dataframe dataframe) {
    final Dataframe predictDataframe =
        isNull(predictorNames) || predictorNames.length == 0 ? getPredictionDataframe(dataframe)
            : dataframe.copy().select(predictorNames);
    var matrix = predictDataframe.toMatrix();
    var allPredictions = getTreePredictions(matrix);
    var predictionMatrix = Nd4j.create(allPredictions, matrix.columns(), matrix.rows());
    List<?> predictions = getFinalPredictions(predictionMatrix);
    return predictDataframe.addColumn(predictionColumnName, predictions);
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

  public RandomForestLearning setShuffle(boolean shuffle) {
    super.setShuffle(shuffle);
    return this;
  }

  private Dataframe getPredictionDataframe(Dataframe dataframe) {
    try {
      return dataframe.copy().mapWithout(responseVariableName);
    } catch (Exception e) {
      return dataframe.copy();
    }
  }
}
