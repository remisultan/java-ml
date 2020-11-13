package org.rsultan.dataframe;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.rsultan.dataframe.printer.DataframePrinter;

import java.util.*;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.regex.Pattern;
import java.util.stream.DoubleStream;
import java.util.stream.Stream;

import static java.lang.Double.parseDouble;
import static java.util.Arrays.stream;
import static java.util.stream.Collectors.toList;
import static java.util.stream.Collectors.toMap;
import static java.util.stream.IntStream.range;

public class Dataframe {

    protected final Map<String, List<?>> data;
    protected final Column<?>[] columns;
    protected final int size;

    public Dataframe(Column<?>[] columns) {
        this.columns = columns;
        this.data = stream(columns).collect(toMap(Column::columnName, Column::values, (e1, e2) -> e1, LinkedHashMap::new));
        var sizes = this.data.values().stream().map(List::size).distinct().collect(toList());
        if (sizes.size() > 1) {
            throw new IllegalArgumentException("Dataframe column values should have the same size");
        }
        this.size = sizes.get(0);
    }

    public <T> Dataframe addColumn(Column<T> column) {
        var newColumn = new Column[]{column};
        return Dataframes.create(
                Stream.of(columns, newColumn).flatMap(Arrays::stream).toArray(Column[]::new)
        );
    }

    public <T> Dataframe withColumn(String columnName, Supplier<T> supplier) {
        var values = range(0, size).boxed().map(num -> supplier.get()).collect(toList());
        return addColumn(new Column<>(columnName, values));
    }

    public <SOURCE, TARGET> Dataframe withColumn(String columnName, String sourceColumn, Function<SOURCE, TARGET> transform) {
        var values = (List<SOURCE>) this.data.get(sourceColumn);
        return addColumn(new Column<>(columnName, values.stream().map(transform).collect(toList())));
    }

    public <SOURCE1, SOURCE2, TARGET> Dataframe withColumn(
            String columnName,
            BiFunction<SOURCE1, SOURCE2, TARGET> transform,
            String sourceColumn1,
            String sourceColumn2
    ) {
        var values1 = (List<SOURCE1>) this.data.get(sourceColumn1);
        var values2 = (List<SOURCE2>) this.data.get(sourceColumn2);
        var targetValues = range(0, values1.size()).boxed()
                .map(index -> transform.apply(values1.get(index), values2.get(index)))
                .collect(toList());
        var newColumn = new Column[]{new Column<>(columnName, targetValues)};
        return Dataframes.create(
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
                .map(stream -> stream.map(Object::toString)
                        .mapToDouble(str -> str.isEmpty() ? 0D : parseDouble(str))
                ).map(DoubleStream::toArray)
                .map(doubles -> Nd4j.create(doubles, doubles.length, 1)).toArray(INDArray[]::new);
        return Nd4j.concat(1, vectorList);
    }

    public void show(int number) {
        DataframePrinter.create(data).print(number);
    }

}



