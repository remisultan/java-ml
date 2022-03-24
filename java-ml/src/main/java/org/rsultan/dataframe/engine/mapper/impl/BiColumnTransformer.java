package org.rsultan.dataframe.engine.mapper.impl;

import static java.lang.String.format;
import static org.rsultan.utils.Constants.NULL_ROW;

import java.util.List;
import java.util.Queue;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.BiFunction;
import org.rsultan.dataframe.Row;
import org.rsultan.dataframe.engine.BaseDataProcessor;
import org.rsultan.dataframe.engine.mapper.BiMapper;
import org.rsultan.dataframe.engine.queue.QueueFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class BiColumnTransformer<T, U, R> extends BaseDataProcessor implements BiMapper<T, U, R> {

  private static final Logger LOG = LoggerFactory.getLogger(BiColumnTransformer.class);

  private final Object column;
  private final Object column2;
  private final BiFunction<T, U, R> function;
  private AtomicInteger columnIndex;
  private AtomicInteger columnIndex2;

  public BiColumnTransformer(Object column, Object column2, BiFunction<T, U, R> function) {
    super(QueueFactory.create());
    this.column = column;
    this.column2 = column2;
    this.function = function;
  }

  @Override
  public void run() {
    try {
      final Queue<Row> queue = QueueFactory.get(queueId);
      while (isStarted()) {
        if (!queue.isEmpty()) {
          final Row row = queue.poll();
          if (!NULL_ROW.equals(row)) {
            final List<R> values = (List<R>) row.values();
            values.set(columnIndex.get(),
                this.map((T) values.get(columnIndex.get()), (U) values.get(columnIndex2.get())));
            this.feed(row);
          } else {
            stop();
          }
        }
      }
    } catch (Exception e) {
      LOG.error("Error while transforming", e);
      stop();
    }
  }

  @Override
  public void start(ExecutorService executor) {
    columnIndex = null;
    columnIndex2 = null;
    super.start(executor);
  }

  @Override
  public void stop() {
    this.feed(NULL_ROW);
    super.stop();
  }

  @Override
  public R map(T element, U element2) {
    return function.apply(element, element2);
  }

  @Override
  public void setHeader(List<Object> header) {
    columnIndex = getColumnIndex(header, column);
    checkIndex(header, columnIndex);
    columnIndex2 = getColumnIndex(header, column2);
    checkIndex(header, columnIndex2);
    super.setHeader(header);
  }

  private void checkIndex(List<Object> header, AtomicInteger columnIndex2) {
    if (columnIndex2.get() == -1) {
      throw new IllegalArgumentException(
          format("Column %s does not exist in %s", column, header));
    }
  }
}
