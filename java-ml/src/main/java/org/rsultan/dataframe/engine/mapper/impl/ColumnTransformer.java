package org.rsultan.dataframe.engine.mapper.impl;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.atomic.AtomicInteger;
import org.rsultan.dataframe.Row;
import org.rsultan.dataframe.engine.BaseDataProcessor;
import org.rsultan.dataframe.engine.mapper.Mapper;
import org.rsultan.dataframe.engine.queue.QueueFactory;

import java.util.List;
import java.util.Queue;
import java.util.function.Function;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static java.lang.String.format;
import static org.rsultan.utils.Constants.NULL_ROW;

public class ColumnTransformer<T, R> extends BaseDataProcessor implements Mapper<T, R> {

  private static final Logger LOG = LoggerFactory.getLogger(ColumnTransformer.class);
  private final Object column;
  private final Function<T, R> function;
  private AtomicInteger columnIndex;

  public ColumnTransformer(Object column, Function<T, R> function) {
    super(QueueFactory.create());
    this.column = column;
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
            values.set(columnIndex.get(), this.map((T) row.get(columnIndex.get())));
            this.feed(row);
          } else {
            stop();
          }
        }
      }
    } catch (Exception e) {
      LOG.error("Error occurred while processing data", e);
      stop();
    }
  }

  @Override
  public void start(ExecutorService executor) {
    super.start(executor);
  }

  @Override
  public void stop() {
    this.feed(NULL_ROW);
    super.stop();
  }

  @Override
  public R map(T element) {
    return function.apply(element);
  }

  @Override
  public void setHeader(List<Object> header) {
    columnIndex = getColumnIndex(header, column);
    if (columnIndex.get() == -1) {
      throw new IllegalArgumentException(
          format("Column %s does not exist in %s", column, header));
    }
    super.setHeader(header);
  }
}
