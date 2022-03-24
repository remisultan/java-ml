package org.rsultan.dataframe.engine.mapper.impl;

import org.rsultan.dataframe.Row;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.rsultan.dataframe.engine.mapper.RowMapperDataProcessor;

import static java.util.Objects.isNull;
import static java.util.Optional.ofNullable;
import static java.util.stream.Collectors.toList;

public class AddColumnMapper<T, R> extends RowMapperDataProcessor {

  private final Object name;
  private final Object sourceName;
  private final Function<T, R> function;

  private AtomicInteger columnIndex;

  public AddColumnMapper(Object column, Object sourceName, Function<T, R> function) {
    super();
    this.name = column;
    this.sourceName = sourceName;
    this.function = function;
  }

  @Override
  public Row map(Row row) {
    var sourceValue = row.values().get(columnIndex.get());
    var targetValue = function.apply((T) sourceValue);
    return new Row(Stream.of(row.values(), List.of(targetValue)).flatMap(List::stream).collect(toList()));
  }

  @Override
  public void setHeader(List<Object> header) {
    columnIndex = getColumnIndex(header, sourceName);
    super.setHeader(Stream.of(header, List.of(name)).flatMap(List::stream).collect(toList()));
  }
}
