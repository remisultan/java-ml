package org.rsultan.core.tree;

import static java.util.Arrays.stream;
import static java.util.stream.Collectors.toList;
import static java.util.stream.IntStream.range;

import java.util.Collections;
import java.util.List;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.rsultan.core.ModelParameters;
import org.rsultan.core.Trainable;
import org.rsultan.dataframe.Column;
import org.rsultan.dataframe.Dataframe;

public abstract class RandomForestLearning
    extends ModelParameters<RandomForestLearning>
    implements Trainable<RandomForestLearning> {

  protected int numberOfEstimators = 100;
  protected double sampleSizeRatio = 0.25;
  protected int treeDepth = 1;
  protected int sampleFeatures = 0;

  protected List<?> responses;
  protected List<?> featureNames;

  private List<? extends DecisionTreeLearning> trees;

  protected abstract List<?> getResponseValues(Dataframe dataframe);

  protected abstract INDArray buildY(Dataframe dataframe);

  protected abstract DecisionTreeLearning buildDecisionTreeLearning();

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
        .mapToObj(idx -> buildDecisionTreeLearning())
        .map(decisionTreeLearning -> {
          var subRowIndices = getSampleIndices(X.rows(), rowSampleSize);
          var subFeatureIndices = getSampleIndices(X.columns(), featureSampleSize);
          var Xsampled = X.getRows(subRowIndices).getColumns(subFeatureIndices);
          var Ysampled = Y.getRows(subRowIndices);
          var localFeatures = stream(subFeatureIndices).mapToObj(featureIndices::get)
              .collect(toList());
          return decisionTreeLearning.train(Xsampled, Ysampled, localFeatures, responses);
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

  public RandomForestLearning setNumberOfEstimators(int numberOfEstimators) {
    this.numberOfEstimators = numberOfEstimators;
    return this;
  }

  public RandomForestLearning setSampleSizeRatio(double sampleSizeRatio) {
    this.sampleSizeRatio = sampleSizeRatio;
    return this;
  }

  public RandomForestLearning setTreeDepth(int treeDepth) {
    this.treeDepth = treeDepth;
    return this;
  }

  public RandomForestLearning setSampleFeatures(int sampleFeatures) {
    this.sampleFeatures = sampleFeatures;
    return this;
  }
}
