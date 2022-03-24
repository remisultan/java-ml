package org.rsultan.core.tree;

import static org.assertj.core.api.AssertionsForInterfaceTypes.assertThat;
import static org.junit.jupiter.params.provider.Arguments.of;
import static org.rsultan.core.ModelSerdeTestUtils.serdeTrainable;
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

public class DecisionTreeLearningTest {

  static {
    Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);
  }

  private static Stream<Arguments> params_that_must_perform_decision_tree_classifier() {
    return Stream.of(
        of(new DecisionTreeClassifier(-1, GINI), new String[]{"y", "x", "x2", "x3"},
            new String[]{"a", "a", "b", "b", "b"}),
        of(new DecisionTreeClassifier(0, GINI), new String[]{"y", "x", "x2", "x3"},
            new String[]{"a", "a", "b", "b", "b"}),
        of(new DecisionTreeClassifier(1, GINI), new String[]{"y", "x", "x2", "x3"},
            new String[]{"a", "a", "b", "b", "b"}),
        of(new DecisionTreeClassifier(2, GINI), new String[]{"y", "x", "x2", "x3"},
            new String[]{"a", "a", "b", "b", "e"}),
        of(new DecisionTreeClassifier(-1, ENTROPY), new String[]{"y", "x", "x2", "x3"},
            new String[]{"a", "a", "b", "b", "b"}),
        of(new DecisionTreeClassifier(0, ENTROPY), new String[]{"y", "x", "x2", "x3"},
            new String[]{"a", "a", "b", "b", "b"}),
        of(new DecisionTreeClassifier(1, ENTROPY), new String[]{"y", "x", "x2", "x3"},
            new String[]{"a", "a", "b", "b", "b"}),
        of(new DecisionTreeClassifier(2, ENTROPY), new String[]{},
            new String[]{"a", "a", "b", "b", "e"})
    );
  }

  private static Stream<Arguments> params_that_must_perform_decision_tree_regressor() {
    return Stream.of(
        of(new DecisionTreeRegressor(-1)),
        of(new DecisionTreeRegressor(0)),
        of(new DecisionTreeRegressor(1)),
        of(new DecisionTreeRegressor(2))
    );
  }

  @ParameterizedTest
  @MethodSource("params_that_must_perform_decision_tree_classifier")
  public void must_perform_decision_tree_classifier(DecisionTreeClassifier decisionTreeClassifier,
      String[] predictorNames,
      String[] expected
  ) throws IOException {
    var dataframe = Dataframes.csv(getResourceFileName("org/rsultan/utils/example-classif.csv"));
    var predictions = decisionTreeClassifier
        .setResponseVariableName("strColumn")
        .setPredictorNames(predictorNames)
        .setShuffle(true)
        .train(dataframe)
        .predict(dataframe)
        .getColumn("predictions");

    assertThat(predictions).containsExactly(expected);

  }

  @ParameterizedTest
  @MethodSource("params_that_must_perform_decision_tree_classifier")
  public void must_serde_and_perform_decision_tree_classifier(
      DecisionTreeClassifier decisionTreeClassifier,
      String[] predictorNames,
      String[] expected
  ) throws IOException {
    var dataframe = Dataframes.csv(getResourceFileName("org/rsultan/utils/example-classif.csv"));
    var predictions = serdeTrainable(decisionTreeClassifier
        .setResponseVariableName("strColumn")
        .setPredictorNames(predictorNames)
        .train(dataframe))
        .predict(dataframe)
        .getColumn("predictions");

    assertThat(predictions).containsExactly(expected);
  }

  @ParameterizedTest
  @MethodSource("params_that_must_perform_decision_tree_regressor")
  public void must_perform_decision_tree_regressor(
      DecisionTreeRegressor decisionTreeRegressor
  )
      throws IOException {
    var dataframe = Dataframes.csv(getResourceFileName("org/rsultan/utils/example-classif.csv"));
    var predictions = decisionTreeRegressor
        .setResponseVariableName("y")
        .setPredictorNames("x", "x2", "x3")
        .train(dataframe)
        .predict(dataframe)
        .getColumn("predictions");

    assertThat(predictions).containsExactly(3.0, 3.0, 3.0, 3.0, 3.0);
  }

  @ParameterizedTest
  @MethodSource("params_that_must_perform_decision_tree_regressor")
  public void must_serde_perform_decision_tree_regressor(
      DecisionTreeRegressor decisionTreeRegressor
  ) throws IOException {
    var dataframe = Dataframes.csv(getResourceFileName("org/rsultan/utils/example-classif.csv"));
    var predictions = serdeTrainable(decisionTreeRegressor
        .setResponseVariableName("y")
        .setPredictorNames("x", "x2", "x3")
        .train(dataframe))
        .predict(dataframe)
        .getColumn("predictions");

    assertThat(predictions).containsExactly(3.0, 3.0, 3.0, 3.0, 3.0);
  }
}
