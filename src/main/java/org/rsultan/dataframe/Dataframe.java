package org.rsultan.dataframe;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.rsultan.dataframe.printer.DataframePrinter;

import java.util.*;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.stream.DoubleStream;
import java.util.stream.Stream;

import static java.lang.Double.parseDouble;
import static java.lang.Math.max;
import static java.lang.Math.min;
import static java.util.Arrays.stream;
import static java.util.Comparator.comparing;
import static java.util.function.Predicate.not;
import static java.util.stream.Collectors.toList;
import static java.util.stream.Collectors.toMap;
import static java.util.stream.IntStream.range;

public class Dataframe {

    public static final String NUMBER_REGEX = "^\\d+(\\.\\d+)*$";
    protected final Map<String, List<?>> data;
    protected final Column<?>[] columns;
    protected final int rows;

    Dataframe(Column<?>[] columns) {
        this.columns = columns;
        this.data = stream(columns).collect(toMap(Column::columnName, Column::values, (e1, e2) -> e1, LinkedHashMap::new));
        var sizes = this.data.values().stream().map(List::size).distinct().collect(toList());
        if (sizes.size() > 1) {
            throw new IllegalArgumentException("Dataframe column values should have the same size");
        }
        this.rows = !sizes.isEmpty() ? sizes.get(0) : 0;
    }

    public <T> Dataframe addColumn(Column<T> column) {
        var newColumn = new Column[]{column};
        return Dataframes.create(
                Stream.of(columns, newColumn).flatMap(Arrays::stream).toArray(Column[]::new)
        );
    }

    public <T> Dataframe addColumns(Column<T>... columns) {
        return Dataframes.create(
                Stream.of(this.columns, columns).flatMap(Arrays::stream).toArray(Column[]::new)
        );
    }

    public <T> Dataframe withColumn(String columnName, Supplier<T> supplier) {
        var values = range(0, rows).boxed().map(num -> supplier.get()).collect(toList());
        return addColumn(new Column<>(columnName, values));
    }

    public Dataframe withoutColumn(String... columnNames) {
        var colList = List.of(columnNames);
        return Dataframes.create(
                stream(columns).filter(column -> !colList.contains(column.columnName())).toArray(Column[]::new)
        );
    }

    public <SOURCE, TARGET> Dataframe withColumn(String columnName, String sourceColumn, Function<SOURCE, TARGET> transform) {
        List<SOURCE> values = this.get(sourceColumn);
        return addColumn(new Column<>(columnName, values.stream().map(transform).collect(toList())));
    }

    public Dataframe oneHotEncode(String columnToEncode) {
        var toEncodeColumnMap = data.get(columnToEncode).stream()
                .distinct().sorted()
                .map(colName -> new Column<Boolean>(colName.toString(), new ArrayList<>()))
                .collect(toMap(Column::columnName, c -> c));

        data.get(columnToEncode).stream().map(Object::toString).forEach(colName -> {
            var trueCol = toEncodeColumnMap.get(colName);
            trueCol.values().add(true);
            toEncodeColumnMap.keySet().stream()
                    .map(toEncodeColumnMap::get)
                    .filter(not(trueCol::equals))
                    .forEach(column -> column.values().add(false));
        });

        var columnArray = toEncodeColumnMap.values().stream().sorted(comparing(Column::columnName)).toArray(Column[]::new);
        return Dataframes.create(
                Stream.of(this.columns, columnArray).flatMap(Arrays::stream).toArray(Column[]::new)
        );
    }

    public <SOURCE1, SOURCE2, TARGET> Dataframe withColumn(
            String columnName,
            BiFunction<SOURCE1, SOURCE2, TARGET> transform,
            String sourceColumn1,
            String sourceColumn2
    ) {
        List<SOURCE1> values1 = this.get(sourceColumn1);
        List<SOURCE2> values2 = this.get(sourceColumn2);
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
        var colNameStream = columnNames.length == 0 ?
                stream(this.columns).map(Column::values) :
                Stream.of(columnNames).map(this::get);

        var vectorList = colNameStream
                .map(List::stream)
                .map(valueStream -> valueStream.mapToDouble(this::objectToDouble))
                .map(DoubleStream::toArray)
                .map(doubles -> Nd4j.create(doubles, doubles.length, 1))
                .toArray(INDArray[]::new);
        return Nd4j.concat(1, vectorList);
    }

    private Double objectToDouble(Object obj) {
        if (obj instanceof Number number) {
            return number.doubleValue();
        } else if (obj instanceof Boolean b) {
            return b ? 1.0D : 0.0D;
        } else if (obj instanceof String s && s.trim().matches(NUMBER_REGEX)) {
            return parseDouble(s.trim());
        } else {
            return (double) String.valueOf(obj).hashCode();
        }
    }

    public void show(int number) {
        this.show(0, number);
    }

    public void show(int start, int end) {
        DataframePrinter.create(data).print(max(0, start), min(end, this.rows));
    }

    public void tail() {
        show(this.rows - 10, this.rows);
    }

    public <T> List<T> get(String column) {
        return List.copyOf((List<T>) data.get(column));
    }

    public int getColumns() {
        return columns.length;
    }

    public int getRows() {
        return rows;
    }
}



