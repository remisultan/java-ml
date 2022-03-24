package org.rsultan.dataframe.engine.mapper.impl;

import static java.util.stream.Collectors.toList;

import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Stream;
import org.rsultan.dataframe.Row;
import org.rsultan.dataframe.engine.mapper.RowMapperDataProcessor;

public class AddColumnFromList extends RowMapperDataProcessor {

  private final Object columnName;
  private final List<Object> values;
  private AtomicInteger atomicInteger;

  public AddColumnFromList(Object columnName, List<Object> values) {
    this.columnName = columnName;
    this.values = values;
  }


  @Override
  public Row map(Row element) {
    var value = atomicInteger.get() < values.size() ? values.get(atomicInteger.getAndIncrement()) : null;
    var values = (List<Object>) element.values();
    values.add(value);
    return new Row(values);
  }

  @Override
  public void start(ExecutorService executorService) {
    super.start(executorService);
  }

  @Override
  public void setHeader(List<Object> header) {
    atomicInteger = new AtomicInteger(0);
    super.setHeader(Stream.of(header, List.of(columnName)).flatMap(List::stream).collect(toList()));
  }
}
