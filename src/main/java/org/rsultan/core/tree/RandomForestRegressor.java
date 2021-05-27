package org.rsultan.core.tree;

import static java.lang.Math.max;
import static java.util.Arrays.stream;
import static java.util.stream.Collectors.toList;
import static org.nd4j.common.util.MathUtils.round;

import java.util.List;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.rsultan.core.tree.domain.Node;
import org.rsultan.dataframe.Dataframe;

public class RandomForestRegressor extends RandomForestLearning {

  public RandomForestRegressor(int numberOfEstimator) {
    super(numberOfEstimator);
  }

  @Override
  protected List<?> getResponseValues(Dataframe dataframe) {
    return dataframe.get(responseVariableName);
  }

  @Override
  protected INDArray buildY(Dataframe dataframe) {
    return dataframe.toMatrix(responseVariableName);
  }

  @Override
  protected DecisionTreeLearning buildDecisionTreeLearning() {
    return new RandomForestRegressorTree(treeDepth, featureNames)
        .setResponseVariableName(responseVariableName)
        .setPredictionColumnName(predictionColumnName);
  }

  @Override
  protected int getFeatureSampleSize(int numberOfFeatures) {
    return numberOfFeatures == 0 ? round(numberOfFeatures / 3.0D) : max(2, numberOfFeatures);
  }

  @Override
  protected List<?> getFinalPredictions(INDArray predictionMatrix) {
    double[] bestResponses = predictionMatrix.mean(true, 0).toDoubleVector();
    return stream(bestResponses).boxed().collect(toList());
  }

  private static class RandomForestRegressorTree extends DecisionTreeRegressor {

    private final List<?> parentFeatureNames;

    public RandomForestRegressorTree(
        int depth,
        List<?> parentFeatureNames) {
      super(depth);
      this.parentFeatureNames = parentFeatureNames;
    }

    @Override
    protected Object getPredictionNodeFeatureName(Node node) {
      return parentFeatureNames.get((int) features.get(node.feature()));
    }

    @Override
    protected Object getNodePrediction(Node node) {
      return node.predictedResponse();
    }
  }
}
