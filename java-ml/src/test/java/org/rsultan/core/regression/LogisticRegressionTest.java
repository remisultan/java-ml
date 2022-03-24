package org.rsultan.core.regression;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.params.provider.Arguments.of;
import static org.rsultan.core.ModelSerdeTestUtils.serdeTrainable;
import static org.rsultan.core.regularization.Regularization.LASSO;
import static org.rsultan.core.regularization.Regularization.NONE;
import static org.rsultan.core.regularization.Regularization.RIDGE;
import static org.rsultan.utils.TestUtils.getResourceFileName;

import java.io.IOException;
import java.nio.file.Files;
import java.util.stream.Stream;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;
import org.rsultan.core.regression.impl.LogisticRegression;
import org.rsultan.core.regularization.Regularization;
import org.rsultan.dataframe.Dataframes;

public class LogisticRegressionTest {

  static {
    Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);
  }

  private static Stream<Arguments> params_that_must_apply_logistic_regression() {
    return Stream.of(
        of("strColumn", "a", NONE, new String[]{"x"},
            new String[]{"a", "Not a", "Not a", "Not a", "Not a"}),
        of("strColumn", "a", RIDGE, new String[]{"x"},
            new String[]{"Not a", "Not a", "Not a", "Not a", "Not a"}),
        of("strColumn", "a", LASSO, new String[]{"x"},
            new String[]{"a", "Not a", "Not a", "Not a", "Not a"}),
        of("strColumn", "a", NONE, new String[]{"x", "x2"},
            new String[]{"a", "a", "Not a", "Not a", "Not a"}),
        of("strColumn", "a", RIDGE, new String[]{"x", "x2"},
            new String[]{"a", "Not a", "Not a", "Not a", "Not a"}),
        of("strColumn", "a", LASSO, new String[]{"x", "x2"},
            new String[]{"Not a", "Not a", "Not a", "Not a", "Not a"}),
        of("strColumn", "a", NONE, new String[]{"x", "x2", "x3"},
            new String[]{"a", "a", "Not a", "Not a", "Not a"}),
        of("strColumn", "a", RIDGE, new String[]{"x", "x2", "x3"},
            new String[]{"a", "Not a", "Not a", "Not a", "Not a"}),
        of("strColumn", "a", LASSO, new String[]{"x", "x2", "x3"},
            new String[]{"a", "Not a", "Not a", "Not a", "Not a"})
    );
  }

  @ParameterizedTest
  @MethodSource("params_that_must_apply_logistic_regression")
  public void must_apply_logistic_regression(
      String responseVariable,
      String label,
      Regularization regularization,
      String[] predictors,
      String[] expectedPredictions
  ) throws IOException {
    var dataframe = Dataframes.csv(getResourceFileName("org/rsultan/utils/example-classif.csv"));
    var logisticRegression = new LogisticRegression(100, 0.5)
        .setPredictorNames(predictors)
        .setResponseVariableName(responseVariable)
        .setRegularization(regularization)
        .setLambda(0.1)
        .setPredictionColumnName("predictions")
        .setChosenLabel(label)
        .setShuffle(true);
    logisticRegression.setLossAccuracyOffset(10);
    logisticRegression.train(dataframe);
    logisticRegression.getHistory().show(10);

    var dfPredict = logisticRegression.predict(dataframe);
    assertThat(dfPredict.<String>getColumn("predictions").stream().toArray())
        .containsExactly(expectedPredictions);
  }

  @ParameterizedTest
  @MethodSource("params_that_must_apply_logistic_regression")
  public void must_serde_and_apply_logistic_regression(
      String responseVariable,
      String label,
      Regularization regularization,
      String[] predictors,
      String[] expectedPredictions
  ) throws IOException {
    var dataframe = Dataframes.csv(getResourceFileName("org/rsultan/utils/example-classif.csv"));
    var logisticRegression = new LogisticRegression(100, 0.5)
        .setPredictorNames(predictors)
        .setResponseVariableName(responseVariable)
        .setRegularization(regularization)
        .setLambda(0.1)
        .setPredictionColumnName("predictions")
        .setChosenLabel(label);
    logisticRegression.setLossAccuracyOffset(10);
    logisticRegression = serdeTrainable(logisticRegression.train(dataframe));
    logisticRegression.getHistory().show(10);

    var dfPredict = logisticRegression.predict(dataframe);
    assertThat(dfPredict.<String>getColumn("predictions").stream().toArray())
        .containsExactly(expectedPredictions);
  }
}
