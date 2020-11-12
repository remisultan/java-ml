package org.rsultan.dataframe;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.regex.Pattern;
import java.util.stream.DoubleStream;
import java.util.stream.Stream;

import static java.lang.Double.parseDouble;
import static java.util.Arrays.stream;
import static java.util.Collections.singletonList;
import static java.util.stream.Collectors.toList;
import static java.util.stream.Collectors.toMap;
import static java.util.stream.IntStream.range;

public final class Dataframe {

    public static final Pattern DOUBLE_VALUE_REGEX = Pattern.compile("\\d+\\.\\d+");
    public static final Pattern LONG_VALUE_REGEX = Pattern.compile("\\d+");
    private final Map<String, List<?>> data;
    private final Column<?>[] columns;
    private final int size;

    private Dataframe(Column<?>[] columns) {
        this.columns = columns;
        this.data = stream(columns).collect(toMap(Column::columnName, Column::values, (e1, e2) -> e1, LinkedHashMap::new));
        var sizes = this.data.values().stream().map(List::size).distinct().collect(toList());
        if (sizes.size() > 1) {
            throw new IllegalArgumentException("Dataframe column values should have the same size");
        }
        this.size = sizes.get(0);
    }

    public static Dataframe create(Column<?>... columns) {
        return new Dataframe(columns);
    }

    public static Dataframe csv(String fileName, String separator, boolean withHeader) throws IOException {
        var path = Paths.get(fileName);
        var reader = Files.newBufferedReader(path);
        var firstLine = reader.readLine().split(separator);
        var columns = range(0, firstLine.length)
                .boxed().map(index -> withHeader ?
                        new Column<>(firstLine[index], new ArrayList<>()) :
                        new Column<>("c".concat(index.toString()), new ArrayList<>(singletonList((getValueWithType(firstLine[index])))))
                ).toArray(Column[]::new);
        reader.lines()
                .map(line -> line.split(separator))
                .forEach(lineArray ->
                        range(0, lineArray.length)
                                .forEach(index -> columns[index].values.add(getValueWithType(lineArray[index])))
                );
        reader.close();
        return new Dataframe(columns);
    }

    private static Object getValueWithType(String value) {
        if (DOUBLE_VALUE_REGEX.matcher(value).matches()) {
            return Double.parseDouble(value);
        } else if (LONG_VALUE_REGEX.matcher(value).matches()) {
            return Long.valueOf(value);
        }
        return value;
    }

    public <T> Dataframe addColumn(Column<T> column) {
        var newColumn = new Column[]{column};
        return Dataframe.create(
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
                .map(stream -> stream.map(Object::toString)
                        .mapToDouble(str -> str.isEmpty() ? 0D : parseDouble(str))
                ).map(DoubleStream::toArray)
                .map(doubles -> Nd4j.create(doubles, doubles.length, 1)).toArray(INDArray[]::new);
        return Nd4j.concat(1, vectorList);
    }

    public void show(int number) {
        DataframePrinter.create(data).print(number);
    }

    public static record Column<T>(String columnName, List<T> values) {
    }
}



