package org.rsultan.core.ensemble.rf;

import static java.lang.Math.sqrt;
import static java.util.Objects.isNull;
import static java.util.stream.Collectors.toList;
import static java.util.stream.IntStream.range;
import static org.nd4j.common.util.MathUtils.round;
import static org.nd4j.linalg.factory.Nd4j.create;

import java.util.List;
import java.util.UUID;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.rsultan.core.tree.DecisionTreeClassifier;
import org.rsultan.core.tree.DecisionTreeLearning;
import org.rsultan.core.tree.domain.Node;
import org.rsultan.core.tree.impurity.ImpurityStrategy;
import org.rsultan.dataframe.Dataframe;

public class RandomForestClassifier extends RandomForestLearning {

  private final ImpurityStrategy impurityStrategy;

  public RandomForestClassifier(int numberOfEstimator, ImpurityStrategy impurityStrategy) {
    super(numberOfEstimator);
    this.impurityStrategy = impurityStrategy;
  }

  @Override
  public RandomForestClassifier train(Dataframe dataframe) {
    super.train(dataframe);
    return this;
  }

  @Override
  public RandomForestClassifier setResponseVariableName(String responseVariableName) {
    super.setResponseVariableName(responseVariableName);
    return this;
  }

  @Override
  public RandomForestClassifier setPredictionColumnName(String name) {
    super.setPredictionColumnName(name);
    return this;
  }

  @Override
  public RandomForestClassifier setPredictorNames(String... names) {
    super.setPredictorNames(names);
    return this;
  }

  @Override
  public RandomForestClassifier setSampleSizeRatio(double sampleSizeRatio) {
    super.setSampleSizeRatio(sampleSizeRatio);
    return this;
  }

  @Override
  public RandomForestClassifier setTreeDepth(int treeDepth) {
    super.setTreeDepth(treeDepth);
    return this;
  }

  @Override
  public RandomForestClassifier setSampleFeatureSize(int sampleFeatures) {
    super.setSampleFeatureSize(sampleFeatures);
    return this;
  }

  @Override
  protected List<?> getResponseValues(Dataframe dataframe) {
    return dataframe.get(responseVariableName).stream().sorted().distinct().collect(toList());
  }

  @Override
  protected INDArray buildY(Dataframe dataframe) {
    if (isNull(responses)) {
      responses = getResponseValues(dataframe);
    }
    var columnTemp = UUID.randomUUID().toString();
    return dataframe.map(columnTemp, responses::indexOf, responseVariableName)
        .toMatrix(columnTemp);
  }

  @Override
  protected DecisionTreeLearning buildDecisionTreeLearning() {
    return new RandomForestClassifierTree(treeDepth, impurityStrategy, this.features,
        this.responses)
        .setResponseVariableName(responseVariableName)
        .setPredictionColumnName(predictionColumnName);
  }

  @Override
  protected int getFeatureSampleSize(int numberOfFeatures) {
    return numberOfFeatures == 0 ? round(sqrt(numberOfFeatures)) : Math.max(2, numberOfFeatures);
  }

  @Override
  protected List<?> getFinalPredictions(INDArray predictionMatrix) {
    return range(0, predictionMatrix.columns()).parallel()
        .mapToObj(predictionMatrix::getColumn)
        .map(label -> range(0, responses.size()).parallel()
            .mapToObj(i -> label.getWhere((double) i, Conditions.equals()))
            .mapToDouble(vector -> vector == null ? 0.0D : vector.columns())
            .toArray())
        .map(doubleArray -> create(doubleArray, doubleArray.length, 1))
        .map(v -> Nd4j.argMax(v).getInt(0, 0))
        .map(responses::get).collect(toList());
  }

  private static class RandomForestClassifierTree extends DecisionTreeClassifier {

    private final List<?> parentFeatureNames;

    public RandomForestClassifierTree(
        int depth,
        ImpurityStrategy strategy,
        List<?> parentFeatureNames,
        List<?> responses

    ) {
      super(depth, strategy);
      this.parentFeatureNames = parentFeatureNames;
      this.responses = responses;
    }

    @Override
    protected Object getPredictionNodeFeatureName(Node node) {
      return parentFeatureNames.get((int) features.get(node.feature()));
    }

    @Override
    protected Object getNodePrediction(Node node) {
      return node.predictedResponse().intValue();
    }
  }
}
