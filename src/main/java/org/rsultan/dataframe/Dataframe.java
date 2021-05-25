package org.rsultan.dataframe;

import static java.lang.Math.max;
import static java.lang.Math.min;
import static java.util.Arrays.stream;
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
import org.nd4j.linalg.api.ndarray.INDArray;
import org.rsultan.dataframe.printer.DataframePrinter;
import org.rsultan.dataframe.transform.filter.FilterDataframe;
import org.rsultan.dataframe.transform.filter.FilterTransform;
import org.rsultan.dataframe.transform.map.MapDataframe;
import org.rsultan.dataframe.transform.map.MapTransform;
import org.rsultan.dataframe.transform.matrix.MatrixDataframe;
import org.rsultan.dataframe.transform.matrix.MatrixTransform;

public class Dataframe implements MapTransform, FilterTransform, MatrixTransform {

  private final Map<?, List<?>> data;
  private final Column<?>[] columns;
  private final int rowSize;

  private final MapTransform mapTransform;
  private final FilterTransform filterTransform;
  private final MatrixTransform matrixTransform;

  Dataframe(Column<?>[] columns) {
    this.columns = columns;
    this.data = stream(columns)
        .collect(toMap(Column::columnName, Column::values, (e1, e2) -> e1, LinkedHashMap::new));
    var sizes = this.data.values().stream().map(List::size).distinct().collect(toList());
    if (sizes.size() > 1) {
      throw new IllegalArgumentException("Dataframe column values should have the same size");
    }
    this.rowSize = !sizes.isEmpty() ? sizes.get(0) : 0;

    this.mapTransform = new MapDataframe(this);
    this.filterTransform = new FilterDataframe(this);
    this.matrixTransform = new MatrixDataframe(this);
  }

  Dataframe(String[] columnNames, Row[] rows) {
    this(getColumnsFromRows(columnNames, rows));
  }

  private static Column<?>[] getColumnsFromRows(String[] columnNames, Row[] rows) {
    var sizes = stream(rows).map(Row::values).map(List::size).distinct().collect(toList());
    if (sizes.size() > 1) {
      throw new IllegalArgumentException("Dataframe row values should have the same size");
    }
    if (columnNames.length != sizes.get(0)) {
      throw new IllegalArgumentException(
          "Dataframe row values should have the same size has the columns");
    }
    return range(0, columnNames.length).mapToObj(idx -> {
      var column = new Column<>(columnNames[idx], new ArrayList<>());
      stream(rows).parallel().map(row -> row.values().get(idx))
          .forEachOrdered(column.values()::add);
      return column;
    }).toArray(Column[]::new);
  }

  public Dataframe select(String... columnNames) {
    return Dataframes.create(
        stream(columnNames)
            .map(colName -> new Column<>(colName, this.get(colName)))
            .toArray(Column[]::new)
    );
  }

  public <SOURCE1> Dataframe filter(String columnName, Predicate<SOURCE1> predicate) {
    return filterTransform.filter(columnName, predicate);
  }

  public <SOURCE1, SOURCE2> Dataframe filter(
      String sourceColumn1,
      String sourceColumn2,
      BiPredicate<SOURCE1, SOURCE2> predicate) {
    return filterTransform.filter(sourceColumn1, sourceColumn2, predicate);
  }

  public <T> Dataframe addColumn(Column<T> column) {
    return Dataframes.create(
        of(columns, new Column[]{column}).flatMap(Arrays::stream).toArray(Column[]::new)
    );
  }

  public <T> Dataframe map(String columnName, Supplier<T> supplier) {
    return mapTransform.map(columnName, supplier);
  }

  public <S, T> Dataframe map(String columnName, Function<S, T> f, String sourceColumn) {
    return mapTransform.map(columnName, f, sourceColumn);
  }

  public <S1, S2, T> Dataframe map(String columnName,
      BiFunction<S1, S2, T> f,
      String sourceColumn1,
      String sourceColumn2) {
    return mapTransform.map(columnName, f, sourceColumn1, sourceColumn2);
  }

  public Dataframe mapWithout(String... columnNames) {
    return mapTransform.mapWithout(columnNames);
  }

  public Dataframe oneHotEncode(String columnToEncode) {
    return matrixTransform.oneHotEncode(columnToEncode);
  }

  public INDArray toVector(String columnName) {
    return matrixTransform.toVector(columnName);
  }

  public INDArray toMatrix(String... columnNames) {
    return matrixTransform.toMatrix(columnNames);
  }

  public void show(int number) {
    this.show(0, number);
  }

  public void show(int start, int end) {
    DataframePrinter.create(data).print(max(0, start), min(end, this.rowSize));
  }

  public void tail() {
    show(this.rowSize - 10, this.rowSize);
  }

  public <T> List<T> get(Object column) {
    return List.copyOf((List<T>) data.get(column));
  }

  public int getColumnSize() {
    return columns.length;
  }

  public Column<?>[] getColumns() {
    return columns;
  }

  public int getRowSize() {
    return rowSize;
  }

  public Map<?, List<?>> getData() {
    return data;
  }
}



