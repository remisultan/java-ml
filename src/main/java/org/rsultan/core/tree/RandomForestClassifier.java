package org.rsultan.core.tree;

import static java.lang.Math.sqrt;
import static java.util.Arrays.stream;
import static java.util.Objects.isNull;
import static java.util.stream.Collectors.toList;
import static org.nd4j.common.util.MathUtils.round;
import static org.rsultan.core.tree.impurity.ImpurityStrategy.GINI;

import java.util.List;
import java.util.UUID;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.rsultan.core.tree.domain.Node;
import org.rsultan.core.tree.impurity.ImpurityStrategy;
import org.rsultan.dataframe.Dataframe;

public class RandomForestClassifier extends RandomForestLearning {

  private ImpurityStrategy impurityStrategy = GINI;

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
    return new RandomForestClassifierTree(treeDepth, GINI, this.featureNames)
        .setResponseVariableName(responseVariableName)
        .setPredictionColumnName(this.predictionColumnName);
  }

  @Override
  protected int getFeatureSampleSize(int numberOfFeatures) {
    return numberOfFeatures == 0 ? round(sqrt(numberOfFeatures)) : Math.max(2, numberOfFeatures);
  }

  @Override
  protected List<?> getFinalPredictions(INDArray predictionMatrix) {
    var impurityService = impurityStrategy.getImpurityService(responses.size());
    var counts = impurityService.getClassCount(predictionMatrix);
    int[] bestResponse = Nd4j.argMax(counts, 1).toIntVector();
    return stream(bestResponse).mapToObj(responses::get).collect(toList());
  }

  public RandomForestClassifier setImpurityStrategy(ImpurityStrategy impurityStrategy) {
    this.impurityStrategy = impurityStrategy;
    return this;
  }

  private static class RandomForestClassifierTree extends DecisionTreeClassifier {

    private final List<?> parentFeatureNames;

    public RandomForestClassifierTree(
        int depth,
        ImpurityStrategy strategy,
        List<?> parentFeatureNames) {
      super(depth, strategy);
      this.parentFeatureNames = parentFeatureNames;
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
