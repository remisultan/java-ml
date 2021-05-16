package org.rsultan.core.tree;

import static org.assertj.core.api.AssertionsForInterfaceTypes.assertThat;
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
        Arguments.of(new DecisionTreeClassifier(-1, GINI), new String[]{"a", "a", "b", "b", "b"}),
        Arguments.of(new DecisionTreeClassifier(0, GINI), new String[]{"a", "a", "b", "b", "b"}),
        Arguments.of(new DecisionTreeClassifier(1, GINI), new String[]{"a", "a", "b", "b", "b"}),
        Arguments.of(new DecisionTreeClassifier(2, GINI), new String[]{"a", "a", "b", "b", "e"}),
        Arguments
            .of(new DecisionTreeClassifier(-1, ENTROPY), new String[]{"a", "a", "b", "b", "b"}),
        Arguments.of(new DecisionTreeClassifier(0, ENTROPY), new String[]{"a", "a", "b", "b", "b"}),
        Arguments.of(new DecisionTreeClassifier(1, ENTROPY), new String[]{"a", "a", "b", "b", "b"}),
        Arguments.of(new DecisionTreeClassifier(2, ENTROPY), new String[]{"a", "a", "b", "b", "e"})
    );
  }

  private static Stream<Arguments> params_that_must_perform_decision_tree_regressor() {
    return Stream.of(
        Arguments.of(new DecisionTreeRegressor(-1)),
        Arguments.of(new DecisionTreeRegressor(0)),
        Arguments.of(new DecisionTreeRegressor(1)),
        Arguments.of(new DecisionTreeRegressor(2))
    );
  }

  @ParameterizedTest
  @MethodSource("params_that_must_perform_decision_tree_classifier")
  public void must_perform_decision_tree_classifier(DecisionTreeClassifier decisionTreeClassifier,
      String[] expected)
      throws IOException {
    var dataframe = Dataframes.csv(getResourceFileName("org/rsultan/utils/example-classif.csv"));
    var predictions = decisionTreeClassifier
        .setResponseVariableName("strColumn")
        .setPredictorNames("y", "x", "x2", "x3")
        .train(dataframe)
        .predict(dataframe)
        .get("predictions");

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
        .get("predictions");

    assertThat(predictions).containsExactly(2.5D, 2.5D, 2.5D, 2.5D, 5.0D);
  }
}
