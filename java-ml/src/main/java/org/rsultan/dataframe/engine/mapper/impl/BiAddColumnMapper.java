package org.rsultan.dataframe.engine.mapper.impl;

import static java.util.stream.Collectors.toList;

import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.BiFunction;
import java.util.stream.Stream;
import org.rsultan.dataframe.Row;
import org.rsultan.dataframe.engine.mapper.RowMapperDataProcessor;

public class BiAddColumnMapper<T, U, R> extends RowMapperDataProcessor {

  private final Object name;
  private final Object sourceName;
  private final Object sourceName2;
  private final BiFunction<T, U, R> function;

  private AtomicInteger columnIndex;
  private AtomicInteger columnIndex2;

  public BiAddColumnMapper(Object column, Object sourceName, Object sourceName2,
      BiFunction<T, U, R> function) {
    super();
    this.name = column;
    this.sourceName = sourceName;
    this.sourceName2 = sourceName2;
    this.function = function;
  }

  @Override
  public Row map(Row row) {
    var sourceValue = row.values().get(columnIndex.get());
    var sourceValue2 = row.values().get(columnIndex.get());
    var targetValue = function.apply((T) sourceValue, (U) sourceValue2);
    return new Row(Stream.of(row.values(), List.of(targetValue)).flatMap(List::stream).collect(toList()));
  }

  @Override
  public void setHeader(List<Object> header) {
    columnIndex = getColumnIndex(header, sourceName);
    columnIndex2 = getColumnIndex(header, sourceName2);
    super.setHeader(Stream.of(header, List.of(name)).flatMap(List::stream).collect(toList()));
  }
}
