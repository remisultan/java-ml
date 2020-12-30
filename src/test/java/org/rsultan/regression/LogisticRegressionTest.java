package org.rsultan.regression;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;
import org.rsultan.dataframe.Dataframes;
import org.rsultan.regression.impl.LogisticRegression;
import org.rsultan.utils.CSVUtilsTest;

import java.io.File;
import java.io.IOException;
import java.util.stream.Stream;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.params.provider.Arguments.of;

public class LogisticRegressionTest {

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
                        "a",
                        new String[]{"x"},
                        new String[]{"a", "Not a", "Not a", "Not a", "Not a"}),
                of("strColumn",
                        "a",
                        new String[]{"x", "x2"},
                        new String[]{"a", "a", "Not a", "Not a", "Not a"}),
                of("strColumn",
                        "a",
                        new String[]{"x", "x2", "x3"},
                        new String[]{"a", "a", "Not a", "Not a", "Not a"})
        );
    }

    @ParameterizedTest
    @MethodSource("params_that_must_apply_softmax_regression")
    public void must_apply_softmax_regression(
            String responseVariable,
            String label,
            String[] predictors,
            String[] expectedPredictions
    ) throws IOException {
        var dataframe = Dataframes.csv(getResourceFileName("org/rsultan/utils/example-classif.csv"));
        var logisticRegression = new LogisticRegression(100, 0.5)
                .setPredictorNames(predictors)
                .setResponseVariableName(responseVariable)
                .setPredictionColumnName("predictions")
                .setLabel(label);
        logisticRegression.setLossAccuracyOffset(10);
        logisticRegression.train(dataframe);
        logisticRegression.getHistory().tail();

        var dfPredict = logisticRegression.predict(dataframe);
        assertThat(dfPredict.<String>get("predictions").stream().toArray()).containsExactly(expectedPredictions);
    }
}
