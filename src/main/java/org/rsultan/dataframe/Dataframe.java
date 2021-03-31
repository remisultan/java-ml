package org.rsultan.dataframe;

import static java.lang.Double.parseDouble;
import static java.lang.Math.max;
import static java.lang.Math.min;
import static java.util.Arrays.stream;
import static java.util.Comparator.comparing;
import static java.util.function.Predicate.not;
import static java.util.stream.Collectors.toList;
import static java.util.stream.Collectors.toMap;
import static java.util.stream.IntStream.range;
import static java.util.stream.Stream.of;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.BiFunction;
import java.util.function.BiPredicate;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.function.Supplier;
import java.util.stream.DoubleStream;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.rsultan.dataframe.printer.DataframePrinter;

public class Dataframe {

  public static final String NUMBER_REGEX = "^\\d+(\\.\\d+)*$";
  protected final Map<String, List<?>> data;
  protected final Column<?>[] columns;
  protected final int rows;

  Dataframe(Column<?>[] columns) {
    this.columns = columns;
    this.data = stream(columns)
        .collect(toMap(Column::columnName, Column::values, (e1, e2) -> e1, LinkedHashMap::new));
    var sizes = this.data.values().stream().map(List::size).distinct().collect(toList());
    if (sizes.size() > 1) {
      throw new IllegalArgumentException("Dataframe column values should have the same size");
    }
    this.rows = !sizes.isEmpty() ? sizes.get(0) : 0;
  }

  public Dataframe select(String... columnNames) {
    var colNameList = List.of(columnNames);
    return Dataframes.create(
        of(columns)
            .filter(column -> colNameList.contains(column.columnName())).toArray(Column[]::new)
    );
  }

  public <SOURCE1> Dataframe filter(String columnName, Predicate<SOURCE1> predicate) {
    List<SOURCE1> values1 = this.get(columnName);
    var indices = range(0, values1.size()).parallel()
        .filter(index -> predicate.test(values1.get(index)))
        .boxed().collect(toList());
    return getFilteredDataframe(indices);
  }

  public <SOURCE1, SOURCE2> Dataframe filter(
      String sourceColumn1,
      String sourceColumn2,
      BiPredicate<SOURCE1, SOURCE2> predicate
  ) {
    List<SOURCE1> values1 = this.get(sourceColumn1);
    List<SOURCE2> values2 = this.get(sourceColumn2);
    var indices = range(0, values1.size()).parallel()
        .filter(index -> predicate.test(values1.get(index), values2.get(index)))
        .boxed().collect(toList());
    return getFilteredDataframe(indices);
  }

  private Dataframe getFilteredDataframe(List<Integer> indices) {
    return Dataframes.create(
        of(columns).map(column -> new Column<>(column.columnName(),
                range(0, column.values().size()).parallel()
                    .filter(indices::contains)
                    .mapToObj(column.values()::get)
                    .collect(toList())
            )
        ).toArray(Column[]::new)
    );
  }

  public <T> Dataframe addColumn(Column<T> column) {
    var newColumn = new Column[]{column};
    return Dataframes.create(
        of(columns, newColumn).flatMap(Arrays::stream).toArray(Column[]::new)
    );
  }

  public <T> Dataframe withColumn(String columnName, Supplier<T> supplier) {
    var values = range(0, rows).boxed().map(num -> supplier.get()).collect(toList());
    return addColumn(new Column<>(columnName, values));
  }

  public Dataframe withoutColumn(String... columnNames) {
    var colList = List.of(columnNames);
    return Dataframes.create(
        stream(columns).filter(column -> !colList.contains(column.columnName()))
            .toArray(Column[]::new)
    );
  }

  public <SOURCE, TARGET> Dataframe withColumn(String columnName, String sourceColumn,
      Function<SOURCE, TARGET> transform) {
    List<SOURCE> values = this.get(sourceColumn);
    return addColumn(new Column<>(columnName, values.stream().map(transform).collect(toList())));
  }

  public <SOURCE1, SOURCE2, TARGET> Dataframe withColumn(
      String columnName,
      BiFunction<SOURCE1, SOURCE2, TARGET> transform,
      String sourceColumn1,
      String sourceColumn2
  ) {
    List<SOURCE1> values1 = this.get(sourceColumn1);
    List<SOURCE2> values2 = this.get(sourceColumn2);
    var targetValues = range(0, values1.size()).parallel().boxed()
        .map(index -> transform.apply(values1.get(index), values2.get(index)))
        .collect(toList());
    var newColumn = new Column[]{new Column<>(columnName, targetValues)};
    return Dataframes.create(
        of(columns, newColumn).flatMap(Arrays::stream).toArray(Column[]::new)
    );
  }

  public Dataframe oneHotEncode(String columnToEncode) {
    var toEncodeColumnMap = data.get(columnToEncode).stream()
        .distinct().sorted()
        .map(colName -> new Column<Boolean>(colName.toString(), new ArrayList<>()))
        .collect(toMap(Column::columnName, c -> c));

    data.get(columnToEncode).parallelStream().map(Object::toString).forEachOrdered(colName -> {
      var trueCol = toEncodeColumnMap.get(colName);
      trueCol.values().add(true);
      toEncodeColumnMap.keySet().parallelStream()
          .map(toEncodeColumnMap::get)
          .filter(not(trueCol::equals))
          .forEachOrdered(column -> column.values().add(false));
    });

    var columnArray = toEncodeColumnMap.values().stream().sorted(comparing(Column::columnName))
        .toArray(Column[]::new);
    return Dataframes.create(
        of(this.columns, columnArray).flatMap(Arrays::stream).toArray(Column[]::new)
    );
  }

  public INDArray toVector(String columnName) {
    double[] doubles = this.data.get(columnName).stream()
        .parallel()
        .mapToDouble(obj -> parseDouble(obj.toString())).toArray();
    return Nd4j.create(doubles, doubles.length, 1);
  }

  public INDArray toMatrix(String... columnNames) {
    var colNameStream = columnNames.length == 0 ?
        stream(this.columns).map(Column::values) :
        of(columnNames).map(this::get);

    var vectorList = colNameStream.parallel()
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
    }
    throw new IllegalArgumentException("Cannot cast " + obj + " to number");
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



