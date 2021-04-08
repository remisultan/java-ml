package org.rsultan.core.regression;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;
import org.rsultan.dataframe.Dataframes;
import org.rsultan.core.regression.impl.SoftmaxRegression;
import org.rsultan.core.regularization.Regularization;
import org.rsultan.utils.CSVUtilsTest;

import java.io.File;
import java.io.IOException;
import java.util.stream.Stream;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.params.provider.Arguments.of;
import static org.rsultan.core.regularization.Regularization.*;

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
                of("strColumn", NONE, new String[]{"x"}, new String[]{"a", "a", "b", "b", "b"}),
                of("strColumn", RIDGE, new String[]{"x"}, new String[]{"a", "b", "b", "b", "b"}),
                of("strColumn", LASSO, new String[]{"x"}, new String[]{"b", "b", "b", "e", "e"}),
                of("strColumn", NONE, new String[]{"x", "x2"}, new String[]{"a", "a", "b", "b", "b"}),
                of("strColumn", RIDGE, new String[]{"x", "x2"}, new String[]{"a", "b", "b", "b", "b"}),
                of("strColumn", LASSO, new String[]{"x", "x2"}, new String[]{"e", "e", "e", "e", "e"}),
                of("strColumn", NONE, new String[]{"x", "x2", "x3"}, new String[]{"a", "a", "b", "b", "b"}),
                of("strColumn", RIDGE, new String[]{"x", "x2", "x3"}, new String[]{"a", "b", "b", "b", "b"}),
                of("strColumn", LASSO, new String[]{"x", "x2", "x3"}, new String[]{"a", "b", "b", "b", "b"})
        );
    }

    @ParameterizedTest
    @MethodSource("params_that_must_apply_softmax_regression")
    public void must_apply_softmax_regression(
            String responseVariable,
            Regularization regularization,
            String[] predictors,
            String[] expectedPredictions
    ) throws IOException {
        var dataframe = Dataframes.csv(getResourceFileName("org/rsultan/utils/example-classif.csv"));
        var softmaxRegression = new SoftmaxRegression(100, 0.1)
                .setPredictorNames(predictors)
                .setResponseVariableName(responseVariable)
                .setRegularization(regularization)
                .setPredictionColumnName("predictions")
                .setLambda(0.1)
                .setLossAccuracyOffset(10)
                .train(dataframe);
        softmaxRegression.getHistory().tail();

        var dfPredict = softmaxRegression.predict(dataframe);
        assertThat(dfPredict.<String>get("predictions").stream().toArray()).containsExactly(expectedPredictions);
    }
}
