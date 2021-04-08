package org.rsultan.utils;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.stream.Stream;

import static java.util.stream.IntStream.rangeClosed;
import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.params.provider.Arguments.of;
import static org.rsultan.utils.Matrices.diagonal;
import static org.rsultan.utils.Matrices.vectorAverage;

public class MatricesTest {

    private static final double[] ONE_NINE_RANGE = rangeClosed(1, 9).asDoubleStream().toArray();
    private static final double[] ONE_TEN_RANGE = rangeClosed(1, 10).asDoubleStream().toArray();
    private static final INDArray MATRIX = Nd4j.create(ONE_TEN_RANGE, 5, 2);
    private static final INDArray ROW_VECTOR = Nd4j.create(ONE_TEN_RANGE, 10, 1);
    private static final INDArray SQUARE_MATRIX = Nd4j.create(ONE_NINE_RANGE, 3, 3);

    static {
        Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);
    }

    private static Stream<Arguments> params_that_must_return_average_vector() {
        return Stream.of(
                of(Nd4j.ones(10, 1), new double[]{1, 0.9999999999999999}),
                of(ROW_VECTOR, new double[]{5.5D, 5.500000000000001})
        );
    }

    private static Stream<Arguments> params_that_must_throw_Exception_due_to_average_input() {
        return Stream.of(
                of(null, NullPointerException.class),
                of(MATRIX, IllegalArgumentException.class)
        );
    }

    private static Stream<Arguments> params_that_must_throw_Exception_due_to_diagonal_input() {
        return Stream.of(
                of(null, NullPointerException.class),
                of(ROW_VECTOR, IllegalArgumentException.class),
                of(MATRIX, IllegalArgumentException.class)
        );
    }

    @Test
    public void must_throw_NullPointerException_due_to_null_params() {
        assertThrows(NullPointerException.class, () -> vectorAverage(null));
    }

    @ParameterizedTest
    @MethodSource("params_that_must_return_average_vector")
    public void must_return_average_matrix(INDArray source, double[] expected) {
        assertThat(vectorAverage(source).toDoubleVector()).containsAnyOf(expected);
    }

    @ParameterizedTest
    @MethodSource("params_that_must_throw_Exception_due_to_average_input")
    public void must_throw_Exception_due_to_average_input(INDArray vector, Class<? extends Exception> exception) {
        assertThrows(exception, () -> vectorAverage(vector));
    }

    @ParameterizedTest
    @MethodSource("params_that_must_throw_Exception_due_to_diagonal_input")
    public void must_throw_Exception_due_to_diagonal_input(INDArray vector, Class<? extends Exception> exception) {
        assertThrows(exception, () -> diagonal(vector));
    }

    @Test
    public void must_return_diagonal_of_matrix() {
        assertThat(diagonal(SQUARE_MATRIX).toDoubleVector()).containsExactly(1.0D, 5.0D, 9.0D);
    }
}
