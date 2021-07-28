package org.rsultan.core.regression;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.params.provider.Arguments.of;
import static org.rsultan.core.ModelSerdeTestUtils.serdeTrainable;
import static org.rsultan.utils.TestUtils.getResourceFileName;

import java.io.IOException;
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
            new double[]{0.6855708946307656, 1.4124415416419548E15},
            new double[]{0.27110493295189886, 0.0},
            new double[]{1.0000000000000016, 2.0000000000000018, 3.0000000000000018,
                4.000000000000002, 5.000000000000002}),
        of("y",
            new String[]{"x", "x2"},
            new double[]{1.1124434706744069E-13, 0.9999999999999791, 1.3183898417423734E-15},
            4.332236737008807E-27,
            6.581972908641304E-14,
            1.0,
            new double[]{0.49839415538353254, 5.878981885019946E12, 0.047400374308657484},
            new double[]{0.3338093937142339, 0.0, 0.4832508422900186},
            new double[]{1.0000000000000917, 2.0000000000000746, 3.000000000000061,
                4.000000000000049, 5.00000000000004}),
        of("y",
            new String[]{"x", "x2", "x3"},
            new double[]{3.3661962106634746E-12, 0.9999999999994151, -1.0902390101819037E-13,
                1.9165224962591765E-14},
            2.498960921826301E-24,
            1.5808102105649181E-12,
            1.0,
            new double[]{0.19358290785808935, 4.3994833288545204E10, -0.01292277119642834,
                0.020574554152200282},
            new double[]{0.43913350496326764, 7.235212429179683E-12, 0.5041132168725599,
                0.49345183987744734},
            new double[]{1.0000000000026914, 2.0000000000019136, 3.000000000001148,
                4.000000000000509, 5.000000000000112})
    );
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
      double[] expectedTValues,
      double[] expectedPValues,
      double[] expectedPredictions
  ) throws IOException {
    var dataframe = Dataframes
        .csv(getResourceFileName("org/rsultan/utils/example-linear-regression.csv"));
    var linearRegression = new LinearRegression()
        .setPredictorNames(predictors)
        .setResponseVariableName(responseVariable)
        .setPredictionColumnName("predictions")
        .train(dataframe);
    linearRegression.showMetrics();

    assertThat(linearRegression.getW().toDoubleVector()).containsExactly(expectedBeta);
    assertThat(round(8, linearRegression.getMSE())).isEqualTo(round(8, expectedMSE));
    assertThat(round(8, linearRegression.getRMSE())).isEqualTo(round(8, expectedRMSE));
    assertThat(round(8, linearRegression.getR2())).isEqualTo(round(8, expectedR2));
    assertThat(linearRegression.gettValues().toDoubleVector()).containsExactly(expectedTValues);
    assertThat(linearRegression.getpValues().toDoubleVector()).containsExactly(expectedPValues);

    var dfPredict = linearRegression.predict(dataframe);
    assertThat(
        dfPredict.<Double>get("predictions").stream().mapToDouble(Double::doubleValue).toArray())
        .containsExactly(expectedPredictions);
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
      double[] expectedTValues,
      double[] expectedPValues,
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

    assertThat(linearRegression.getW().toDoubleVector()).containsExactly(expectedBeta);
    assertThat(round(8, linearRegression.getMSE())).isEqualTo(round(8, expectedMSE));
    assertThat(round(8, linearRegression.getRMSE())).isEqualTo(round(8, expectedRMSE));
    assertThat(round(8, linearRegression.getR2())).isEqualTo(round(8, expectedR2));
    assertThat(linearRegression.gettValues().toDoubleVector()).containsExactly(expectedTValues);
    assertThat(linearRegression.getpValues().toDoubleVector()).containsExactly(expectedPValues);

    var dfPredict = linearRegression.predict(dataframe);
    assertThat(
        dfPredict.<Double>get("predictions").stream().mapToDouble(Double::doubleValue).toArray())
        .containsExactly(expectedPredictions);
  }
}
