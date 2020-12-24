package org.rsultan.regression;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;
import org.rsultan.dataframe.Dataframes;
import org.rsultan.utils.CSVUtilsTest;

import java.io.File;
import java.io.IOException;
import java.util.stream.Stream;

import static java.lang.Double.NaN;
import static java.lang.Double.POSITIVE_INFINITY;
import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.params.provider.Arguments.of;

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
                        new double[]{0.0D, 1.0D},
                        0.0D,
                        0.0D,
                        1.0D,
                        new double[]{NaN, POSITIVE_INFINITY},
                        new double[]{NaN, 1.0D},
                        new double[]{1.0D, 2.0D, 3.0D, 4.0D, 5.0D}),
                of("y",
                        new String[]{"x", "x2"},
                        new double[]{1.1368683772161603E-13, 1.0D, 3.552713678800501E-15},
                        2.428176975142692E-26,
                        1.558260881605738E-13,
                        1.0D,
                        new double[]{0.21513996713395805, 2.4832362766959023E12, 0.05395280807828471},
                        new double[]{0.5751982995600785, 1.0, 0.5190613318378214},
                        new double[]{1.0000000000001172D, 2.000000000000128D, 3.0000000000001457D, 4.0000000000001705D, 5.0000000000002025D}),
                of("y",
                        new String[]{"x", "x2", "x3"},
                        new double[]{3.410605131648481E-12, 0.9999999999995453, -1.1368683772161603E-13, 0.0D},
                        3.2544387225133184E-24,
                        1.8040062978031198E-12,
                        1.0D,
                        new double[]{0.17187024920339392, 3.855168452534568E10, -0.01180825727641034, 0.0D},
                        new double[]{0.5541786697289189, 0.9999999999917433, 0.496241489653043, 0.5D},
                        new double[]{1.0000000000028422D, 2.0000000000020464D, 3.000000000001023D,  3.9999999999997726D, 4.999999999998295D})
        );
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
        var dataframe = Dataframes.csv(getResourceFileName("org/rsultan/utils/example.csv"));
        var linearRegression = new LinearRegression()
                .setPredictorNames(predictors)
                .setResponseVariableName(responseVariable)
                .setPredictionColumnName("predictions")
                .train(dataframe);
        linearRegression.showMetrics();

        assertThat(linearRegression.getBETA().toDoubleVector()).containsExactly(expectedBeta);
        assertThat(linearRegression.getMSE()).isEqualTo(expectedMSE);
        assertThat(linearRegression.getRMSE()).isEqualTo(expectedRMSE);
        assertThat(linearRegression.getR2()).isEqualTo(expectedR2);
        assertThat(linearRegression.gettValues().toDoubleVector()).containsExactly(expectedTValues);
        assertThat(linearRegression.getpValues().toDoubleVector()).containsExactly(expectedPValues);

        var dfPredict = linearRegression.predict(dataframe);
        assertThat(dfPredict.<Double>get("predictions").stream().mapToDouble(Double::doubleValue).toArray()).containsExactly(expectedPredictions);
    }
}
