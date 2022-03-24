package org.rsultan.dataframe.engine.mapper.impl;

import static java.util.stream.Collectors.toCollection;
import static java.util.stream.Collectors.toList;

import java.util.List;
import java.util.SortedSet;
import java.util.TreeSet;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Function;
import java.util.stream.Stream;
import org.rsultan.dataframe.Row;
import org.rsultan.dataframe.engine.mapper.AccumulatorDataProcessor;

public class OneHotEncoderMapper extends AccumulatorDataProcessor {

  private final Object columnName;
  private AtomicInteger columnIndex;
  private SortedSet<Object> columnValues;

  public OneHotEncoderMapper(Object column) {
    super();
    this.columnName = column;
  }

  @Override
  public Row map(Row row) {
    if (columnValues == null){
      columnValues = accumulator.stream().map(Row::values)
          .map(values -> values.get(columnIndex.get()))
          .collect(toCollection(TreeSet::new));
      List<Object> newHeaders = Stream.of(header.stream(), columnValues.stream()).flatMap(Function.identity()).collect(toList());
      super.setHeader(newHeaders);
      super.propagateHeader(newHeaders);
    }

    var value = row.values().get(columnIndex.get());
    var values = (List<Object>) row.values();
    columnValues.stream().map(currVal -> currVal.equals(value) ? 1D : 0D).forEach(values::add);

    return new Row(values);
  }

  @Override
  public void start(ExecutorService executor) {
    super.start(executor);
  }

  @Override
  public void setHeader(List<Object> header) {
    columnIndex = getColumnIndex(header, columnName);
    super.setHeader(header);
  }
}
