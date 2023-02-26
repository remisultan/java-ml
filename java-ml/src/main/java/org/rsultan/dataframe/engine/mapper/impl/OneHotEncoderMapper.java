package org.rsultan.dataframe.engine.mapper.impl;

import static java.util.function.Function.identity;
import static java.util.stream.Collectors.toCollection;
import static java.util.stream.Collectors.toList;

import java.util.ArrayList;
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
  protected List<Row> accumulator;

  public OneHotEncoderMapper(Object column) {
    super();
    this.columnName = column;
  }

  @Override
  protected void accumulate(Row row) {
    accumulator.add(row);
  }

  @Override
  protected void feedFromAccumulator() {
    columnValues = accumulator.stream().map(Row::values)
        .map(values -> values.get(columnIndex.get()))
        .collect(toCollection(TreeSet::new));
    var newHeaders = Stream.of(header.stream(), columnValues.stream())
        .flatMap(identity())
        .collect(toList());
    super.setHeader(newHeaders);
    super.propagateHeader(newHeaders);
    this.accumulator.stream().map(this::map).forEach(this::feed);
  }

  @Override
  public Row map(Row row) {
    var value = row.values().get(columnIndex.get());
    var values = (List<Object>) row.values();
    columnValues.stream().map(currVal -> currVal.equals(value) ? 1D : 0D).forEach(values::add);

    return new Row(values);
  }

  @Override
  public void start(ExecutorService executor) {
    accumulator = new ArrayList<>();
    super.start(executor);
  }

  @Override
  public void setHeader(List<Object> header) {
    columnIndex = getColumnIndex(header, columnName);
    super.setHeader(header);
  }

  @Override
  public void stop() {
    accumulator.clear();
    super.stop();
  }
}
