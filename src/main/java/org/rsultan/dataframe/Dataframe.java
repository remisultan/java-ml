package org.rsultan.dataframe;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import static java.util.stream.Collectors.toList;
import static java.util.stream.Collectors.toMap;

public final class Dataframe {

    private final Map<String, List<?>> data;
    private final Column[] columns;

    private Dataframe(Column[] columns) {
        this.columns = columns;
        this.data = Arrays.stream(columns).collect(toMap(Column::columnName, Column::values));
        long sizes = this.data.values().stream().map(List::size).distinct().count();
        if (sizes > 1) {
            throw new IllegalArgumentException("Dataframe column values should have the same size");
        }
    }

    public static Dataframe create(Column... columns) {
        return new Dataframe(columns);
    }

    public <SOURCE, TARGET> Dataframe withColumn(String columnName, String sourceColumn, Function<SOURCE, TARGET> transform) {
        var values = (List<SOURCE>) this.data.get(sourceColumn);
        var newColumn = new Column[]{new Column(columnName, values.stream().map(transform).collect(toList()))};

        return Dataframe.create(
                Stream.of(newColumn, columns).flatMap(Arrays::stream).toArray(Column[]::new)
        );
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
                Stream.of(newColumn, columns).flatMap(Arrays::stream).toArray(Column[]::new)
        );
    }

    public void show(int number) {
        DataframePrinter.create(data).print(number);
    }

    public static record Column(String columnName, List<?> values) {
    }
}



