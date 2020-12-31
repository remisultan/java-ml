package org.rsultan.regularization;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.stream.Stream;

import static org.assertj.core.api.Assertions.assertThat;

public class RegularizationTest {

    static {
        Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);
    }

    private static Stream<Arguments> params_that_must_regularize_regarding_input() {
        return Stream.of(
                Arguments.of(Regularization.NONE, Nd4j.ones(5, 1), 1E-4, 0, Nd4j.zeros(5, 1)),
                Arguments.of(Regularization.RIDGE, Nd4j.ones(5, 1), 1E-4, 2.5E-4, Nd4j.ones(5, 1).mul(1E-4)),
                Arguments.of(Regularization.LASSO, Nd4j.ones(5, 1), 1E-4, 5.0E-4, Nd4j.ones(5, 1).mul(1E-4))
        );
    }

    @ParameterizedTest
    @MethodSource("params_that_must_regularize_regarding_input")
    public void must_regularize_regarding_input(
            Regularization regularization,
            INDArray weights,
            double lambda,
            double regularizedLoss,
            INDArray regularizedGradient) {
        var regularizer = regularization.getRegularizer(weights, lambda);

        assertThat(regularizer.regularize()).isEqualTo(regularizedLoss);
        assertThat(regularizer.gradientRegularize()).isEqualTo(regularizedGradient);
    }
}
