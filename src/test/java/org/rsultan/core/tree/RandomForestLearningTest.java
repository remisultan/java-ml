package org.rsultan.core.tree;

import static org.assertj.core.api.AssertionsForInterfaceTypes.assertThat;
import static org.junit.jupiter.params.provider.Arguments.of;
import static org.rsultan.core.tree.impurity.ImpurityStrategy.ENTROPY;
import static org.rsultan.core.tree.impurity.ImpurityStrategy.GINI;
import static org.rsultan.utils.TestUtils.getResourceFileName;

import java.io.IOException;
import java.util.stream.Stream;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;
import org.rsultan.dataframe.Dataframes;

public class RandomForestLearningTest {

  static {
    Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);
  }

  private static Stream<Arguments> params_that_must_perform_decision_tree_classifier() {
    return Stream.of(
        of(new RandomForestClassifier(-1, GINI), 1, 1, 0.2),
        of(new RandomForestClassifier(0, GINI), 2, 2, 0.3),
        of(new RandomForestClassifier(50, GINI), 3, 3, 0.4),
        of(new RandomForestClassifier(100, GINI), 5, 5, 0.5),
        of(new RandomForestClassifier(-1, ENTROPY), 1, 1, 0.2),
        of(new RandomForestClassifier(0, ENTROPY), 2, 2, 0.3),
        of(new RandomForestClassifier(5, ENTROPY), 3, 3, 0.4),
        of(new RandomForestClassifier(15, ENTROPY), 5, 5, 0.5)
    );
  }

  private static Stream<Arguments> params_that_must_perform_decision_tree_regressor() {
    return Stream.of(
        of(new RandomForestRegressor(-1), 1, 1, 0.2),
        of(new RandomForestRegressor(0), 2, 2, 0.3),
        of(new RandomForestRegressor(50), 3, 3, 0.4),
        of(new RandomForestRegressor(100), 5, 5, 0.5));
  }

  @ParameterizedTest
  @MethodSource("params_that_must_perform_decision_tree_classifier")
  public void must_perform_decision_tree_classifier(RandomForestClassifier decisionTreeClassifier,
      int treeDepth,
      int sampleFeatures,
      double sampleSizeRatio
  )
      throws IOException {
    var dataframe = Dataframes.csv(getResourceFileName("org/rsultan/utils/example-classif.csv"));
    var predictions = decisionTreeClassifier
        .setResponseVariableName("strColumn")
        .setPredictionColumnName("predictions")
        .setPredictorNames("x", "x2", "x3")
        .setTreeDepth(treeDepth)
        .setSampleFeatureSize(sampleFeatures)
        .setSampleSizeRatio(sampleSizeRatio)
        .train(dataframe)
        .predict(dataframe)
        .get("predictions");

    assertThat(predictions).hasSize(5);
  }

  @ParameterizedTest
  @MethodSource("params_that_must_perform_decision_tree_regressor")
  public void must_perform_decision_tree_regressor(
      RandomForestRegressor decisionTreeRegressor,
      int treeDepth,
      int sampleFeatures,
      double sampleSizeRatio
  ) throws IOException {
    var dataframe = Dataframes.csv(getResourceFileName("org/rsultan/utils/example-classif.csv"));
    var predictions = decisionTreeRegressor
        .setResponseVariableName("y")
        .setPredictionColumnName("predictions")
        .setPredictorNames("x", "x2", "x3")
        .setTreeDepth(treeDepth)
        .setSampleFeatureSize(sampleFeatures)
        .setSampleSizeRatio(sampleSizeRatio)
        .train(dataframe)
        .predict(dataframe)
        .get("predictions");

    assertThat(predictions).hasSize(5);
  }
}
