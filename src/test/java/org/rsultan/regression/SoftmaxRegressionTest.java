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

public class SoftmaxRegressionTest {

    static {
        Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);
    }

    public static String getResourceFileName(String resourcePath) {
        var classLoader = CSVUtilsTest.class.getClassLoader();
        return new File(classLoader.getResource(resourcePath).getFile()).toString();
    }

    private static Stream<Arguments> params_that_must_apply_softmax_regression() {
        return Stream.of(
                of("strColumn",
                        new String[]{"x"},
                        new String[]{"e", "e", "e", "e", "e"}),
                of("strColumn",
                        new String[]{"x", "x2"},
                        new String[]{"a", "a", "a", "a", "a"}),
                of("strColumn",
                        new String[]{"x", "x2", "x3"},
                        new String[]{"a", "a", "a", "a", "a"})
        );
    }

    @ParameterizedTest
    @MethodSource("params_that_must_apply_softmax_regression")
    public void must_apply_softmax_regression(
            String responseVariable,
            String[] predictors,
            String[] expectedPredictions
    ) throws IOException {
        var dataframe = Dataframes.csv(getResourceFileName("org/rsultan/utils/example-classif.csv"));
        var softmaxRegression = new SoftmaxRegression(10, 0.01)
                .setPredictorNames(predictors)
                .setResponseVariableName(responseVariable)
                .setPredictionColumnName("predictions")
                .train(dataframe);
        softmaxRegression.getHistory().tail();

        var dfPredict = softmaxRegression.predict(dataframe);
        assertThat(dfPredict.<String>get("predictions").stream().toArray()).containsExactly(expectedPredictions);
    }
}
