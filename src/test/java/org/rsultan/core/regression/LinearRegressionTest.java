package org.rsultan.core.regression;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;
import org.rsultan.dataframe.Dataframes;
import org.rsultan.core.regression.impl.LinearRegression;
import org.rsultan.utils.CSVUtilsTest;

import java.io.File;
import java.io.IOException;
import java.util.stream.Stream;

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
                        new double[]{-2.506269592476489, 3.136363636363635},
                        0.19435736677115983,
                        0.440859803986664,
                        0.9028213166144201,
                        new double[]{-3.2571720041509487, 9.360368540114692},
                        new double[]{0.02361623147013414, 0.9987087826425575},
                        new double[]{0.6300940438871461, 3.766457680250781, 6.902821316614416, 10.03918495297805, 13.175548589341686}),
                of("y",
                        new String[]{"x", "x2"},
                        new double[]{3.6994152781695675, -5.663780713939161, 2.9897427464590933},
                        0.015929603431057357,
                        0.12621253278124703,
                        0.9920351982844713,
                        new double[]{4.072386773691373, -5.460955300045554, 13.427152773171985},
                        new double[]{0.9723299897191993, 0.015967339982478877, 0.9972495290894307},
                        new double[]{1.0253773106894997, 4.3308248361276185, 13.615757854483924, 28.880176365758416, 50.12408036995109}),
                of("y",
                        new String[]{"x", "x2", "x3"},
                        new double[]{-8.170418390061968, 16.591950365871526, -10.47235921093079, 3.050349984089735},
                        5.727012408526988E-4,
                        0.023931177172314338,
                        0.9997136493795736,
                        new double[]{-5.010158649914121, 6.139000734969618, -8.380896856206016, 18.413660521274267},
                        new double[]{0.06270883150682838, 0.9486009967934703, 0.03780169010687099, 0.9827303480806939},
                        new double[]{0.9995227489685021, 7.526845370675801, 29.713649379598337, 85.86203468027452, 194.27410117724276})
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

        assertThat(linearRegression.getW().toDoubleVector()).containsExactly(expectedBeta);
        assertThat(linearRegression.getMSE()).isEqualTo(expectedMSE);
        assertThat(linearRegression.getRMSE()).isEqualTo(expectedRMSE);
        assertThat(linearRegression.getR2()).isEqualTo(expectedR2);
        assertThat(linearRegression.gettValues().toDoubleVector()).containsExactly(expectedTValues);
        assertThat(linearRegression.getpValues().toDoubleVector()).containsExactly(expectedPValues);

        var dfPredict = linearRegression.predict(dataframe);
        assertThat(dfPredict.<Double>get("predictions").stream().mapToDouble(Double::doubleValue).toArray()).containsExactly(expectedPredictions);
    }
}
