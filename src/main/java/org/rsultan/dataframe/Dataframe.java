package org.rsultan.dataframe;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import static java.lang.Double.parseDouble;
import static java.util.stream.Collectors.toList;
import static java.util.stream.Collectors.toMap;

public final class Dataframe {

    private final Map<String, List<?>> data;
    private final Column[] columns;
    private final int size;

    private Dataframe(Column[] columns) {
        this.columns = columns;
        this.data = Arrays.stream(columns).collect(toMap(Column::columnName, Column::values, (e1, e2) -> e1, LinkedHashMap::new));
        var sizes = this.data.values().stream().map(List::size).distinct().collect(toList());
        if (sizes.size() > 1) {
            throw new IllegalArgumentException("Dataframe column values should have the same size");
        }
        this.size = sizes.get(0);
    }

    public static Dataframe create(Column... columns) {
        return new Dataframe(columns);
    }

    public Dataframe addColumn(Column column) {
        var newColumn = new Column[]{column};
        return Dataframe.create(
                Stream.of(columns, newColumn).flatMap(Arrays::stream).toArray(Column[]::new)
        );
    }

    public <T> Dataframe withColumn(String columnName, Supplier<T> supplier) {
        var values = IntStream.range(0, size).boxed().map(num -> supplier.get()).collect(toList());
        return addColumn(new Column(columnName, values));
    }

    public <SOURCE, TARGET> Dataframe withColumn(String columnName, String sourceColumn, Function<SOURCE, TARGET> transform) {
        var values= (List<SOURCE>) this.data.get(sourceColumn);
        return addColumn(new Column(columnName, values.stream().map(transform).collect(toList())));
    }

    public <SOURCE1, SOURCE2, TARGET> Dataframe withColumn(
            String columnName,
            BiFunction<SOURCE1, SOURCE2, TARGET> transform,
            String sourceColumn1,
            String sourceColumn2
    ) {
        var values1 = (List<SOURCE1>) this.data.get(sourceColumn1);
        var values2 = (List<SOURCE2>) this.data.get(sourceColumn2);
        var targetValues = IntStream.range(0, values1.size()).boxed()
                .map(index -> transform.apply(values1.get(index), values2.get(index)))
                .collect(toList());
        var newColumn = new Column[]{new Column(columnName, targetValues)};
        return Dataframe.create(
                Stream.of(columns, newColumn).flatMap(Arrays::stream).toArray(Column[]::new)
        );
    }

    public INDArray toVector(String columnName) {
        double[] doubles = this.data.get(columnName).stream().mapToDouble(obj -> parseDouble(obj.toString())).toArray();
        return Nd4j.create(doubles, doubles.length, 1);
    }

    public INDArray toMatrix(String... columnNames) {
        var vectorList = Stream.of(columnNames).sorted()
                .map(this.data::get)
                .map(List::stream)
                .map(stream -> stream.mapToDouble(obj -> parseDouble(obj.toString())).toArray())
                .map(doubles -> Nd4j.create(doubles, doubles.length, 1)).toArray(INDArray[]::new);
        return Nd4j.concat(1, vectorList);
    }

    public void show(int number) {
        DataframePrinter.create(data).print(number);
    }

    public static record Column(String columnName, List<?> values) {
    }
}



