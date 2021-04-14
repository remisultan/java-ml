package org.rsultan.core.regression;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.params.provider.Arguments.of;

import java.io.File;
import java.io.IOException;
import java.util.stream.Stream;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;
import org.rsultan.core.regression.impl.LinearRegression;
import org.rsultan.dataframe.Dataframes;
import org.rsultan.utils.CSVUtilsTest;

public class LinearRegressionTest {

  static {
    Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);
  }

  public static String getResourceFileName(String resourcePath) {
    var classLoader = CSVUtilsTest.class.getClassLoader();
    return new File(classLoader.getResource(resourcePath).getFile()).toString();
  }

  private static Stream<Arguments> params_that_must_apply_linear_regression() {
    return Stream.of(
        of("y",
            new String[]{"x"},
            new double[]{-2.5062695924764875, 3.136363636363635},
            0.19435736677115964,
            0.4408598039866638,
            0.9028213166144202,
            new double[]{-3.257172004150948, 9.360368540114695},
            new double[]{0.9763837685298659, 0.0012912173574425312},
            new double[]{0.6300940438871474, 3.7664576802507823, 6.902821316614418,
                10.039184952978053, 13.175548589341687}),
        of("y",
            new String[]{"x", "x2"},
            new double[]{3.6994152781695915, -5.663780713939179, 2.9897427464590947},
            0.015929603431057288,
            0.12621253278124675,
            0.9920351982844714,
            new double[]{4.072386773691408, -5.460955300045582, 13.427152773172018},
            new double[]{0.027670010280800295, 0.9840326600175213, 0.002750470910569236},
            new double[]{1.0253773106895072, 4.330824836127612, 13.615757854483908,
                28.88017636575839, 50.124080369951066}),
        of("y",
            new String[]{"x", "x2", "x3"},
            new double[]{-8.170418390061572, 16.59195036587023, -10.47235921093084,
                3.050349984089662},
            5.727012408526747E-4,
            0.023931177172313835,
            0.9997136493795736,
            new double[]{-5.010158649913984, 6.139000734969269, -8.380896856206233,
                18.413660521274217},
            new double[]{0.93729116849317, 0.051399003206532456, 0.9621983098931299,
                0.017269651919306184},
            new double[]{0.999522748967479, 7.526845370672824, 29.713649379592425,
                85.86203468026427, 194.27410117722633})
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
    var dataframe = Dataframes.csv(getResourceFileName("org/rsultan/utils/example-linear-regression.csv"));
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
}
