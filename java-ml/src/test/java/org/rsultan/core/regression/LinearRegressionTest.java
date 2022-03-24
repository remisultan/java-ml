package org.rsultan.core.regression;

import static java.lang.Double.NaN;
import static java.lang.Double.POSITIVE_INFINITY;
import static java.util.Arrays.stream;
import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.params.provider.Arguments.of;
import static org.rsultan.core.ModelSerdeTestUtils.serdeTrainable;
import static org.rsultan.utils.TestUtils.getResourceFileName;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.OptionalDouble;
import java.util.stream.Stream;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;
import org.rsultan.core.ModelSerdeTestUtils;
import org.rsultan.core.regression.impl.LinearRegression;
import org.rsultan.dataframe.Dataframes;

public class LinearRegressionTest {

  static {
    Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);
  }

  private static Stream<Arguments> params_that_must_apply_linear_regression() {
    return Stream.of(
        of("y",
            new String[]{"x"},
            new double[]{1.609823385706477E-15, 1.0},
            3.0075322011551075E-30,
            1.7342238036525468E-15,
            1.0,
            new double[]{1.0000000000000016, 2.0000000000000018, 3.0000000000000018,
                4.000000000000002, 5.000000000000002}),
        of("y",
            new String[]{"x", "x2"},
            new double[]{1.1124434706744069E-13, 0.9999999999999791, 1.3183898417423734E-15},
            4.332236737008807E-27,
            6.581972908641304E-14,
            1.0,
            new double[]{1.0000000000000917, 2.0000000000000746, 3.000000000000061,
                4.000000000000049, 5.00000000000004}),
        of("y",
            new String[]{"x", "x2", "x3"},
            new double[]{3.3661962106634746E-12, 0.9999999999994151, -1.0902390101819037E-13,
                1.9165224962591765E-14},
            2.498960921826301E-24,
            1.5808102105649181E-12,
            1.0,
            new double[]{1.0000000000026914, 2.0000000000019136, 3.000000000001148,
                4.000000000000509, 5.000000000000112})
    );
  }

  private static double[] roundAll(double exponent, double[] value) {
    return stream(value).map(d -> round(exponent, d)).toArray();
  }

  private static double round(double exponent, double value) {
    var tenthPower = Math.pow(10, exponent);
    return Math.round(value * tenthPower) / tenthPower;
  }

  @ParameterizedTest
  @MethodSource("params_that_must_apply_linear_regression")
  public void must_apply_linear_regression(
      String responseVariable,
      String[] predictors,
      double[] expectedBeta,
      double expectedMSE,
      double expectedRMSE,
      double expectedR2,
      double[] expectedPredictions
  ) throws IOException {
    var dataframe = Dataframes
        .csv(getResourceFileName("org/rsultan/utils/example-linear-regression.csv"));
    var linearRegression = new LinearRegression()
        .setPredictorNames(predictors)
        .setResponseVariableName(responseVariable)
        .setPredictionColumnName("predictions")
        .setShuffle(true);

    linearRegression.train(dataframe);
    linearRegression.showMetrics();

    assertThat(roundAll(8, linearRegression.getW().toDoubleVector())).containsExactly(roundAll(8, expectedBeta));
    assertThat(round(8, linearRegression.getMSE())).isEqualTo(round(8, expectedMSE));
    assertThat(round(8, linearRegression.getRMSE())).isEqualTo(round(8, expectedRMSE));
    assertThat(round(8, linearRegression.getR2())).isEqualTo(round(8, expectedR2));
    assertThat(linearRegression.gettValues().toDoubleVector()).isNotEmpty();
    assertThat(linearRegression.getpValues().toDoubleVector()).isNotEmpty();

    var dfPredict = linearRegression.predict(dataframe);
    assertThat(
        dfPredict.<Double>getColumn("predictions").stream().mapToDouble(d -> round(8, d)).toArray())
        .containsExactly(roundAll(8, expectedPredictions));
  }

  @ParameterizedTest
  @MethodSource("params_that_must_apply_linear_regression")
  public void must_serde_and_apply_linear_regression(
      String responseVariable,
      String[] predictors,
      double[] expectedBeta,
      double expectedMSE,
      double expectedRMSE,
      double expectedR2,
      double[] expectedPredictions
  ) throws IOException {
    var dataframe = Dataframes
        .csv(getResourceFileName("org/rsultan/utils/example-linear-regression.csv"));
    var linearRegression = serdeTrainable(new LinearRegression()
        .setPredictorNames(predictors)
        .setResponseVariableName(responseVariable)
        .setPredictionColumnName("predictions")
        .train(dataframe));
    linearRegression.showMetrics();

    assertThat(roundAll(8, linearRegression.getW().toDoubleVector())).containsExactly(
        roundAll(8, expectedBeta));
    assertThat(round(8, linearRegression.getMSE())).isEqualTo(round(8, expectedMSE));
    assertThat(round(8, linearRegression.getRMSE())).isEqualTo(round(8, expectedRMSE));
    assertThat(round(8, linearRegression.getR2())).isEqualTo(round(8, expectedR2));
    assertThat(linearRegression.gettValues().toDoubleVector()).isNotEmpty();
    assertThat(linearRegression.getpValues().toDoubleVector()).isNotEmpty();

    var dfPredict = linearRegression.predict(dataframe);
    assertThat(
        dfPredict.<Double>getColumn("predictions").stream().mapToDouble(d -> round(8, d)).toArray())
        .containsExactly(roundAll(8, expectedPredictions));
  }
}
