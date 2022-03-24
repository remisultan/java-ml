package org.rsultan.dataframe.engine.mapper.impl;

import static java.lang.String.format;
import static java.util.stream.Collectors.toList;
import static org.rsultan.utils.Constants.NULL_ROW;

import com.sun.jdi.VoidType;
import java.util.List;
import java.util.Queue;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.rsultan.dataframe.Row;
import org.rsultan.dataframe.engine.BaseDataProcessor;
import org.rsultan.dataframe.engine.mapper.Mapper;
import org.rsultan.dataframe.engine.queue.QueueFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public class SupplierRowMapper<T> extends BaseDataProcessor implements Mapper<Void, T> {

  private final Logger LOG = LoggerFactory.getLogger(SupplierRowMapper.class);
  private final Object column;
  private final Supplier<T> supplier;

  public SupplierRowMapper(Object column, Supplier<T> supplier) {
    super(QueueFactory.create());
    this.column = column;
    this.supplier = supplier;
  }

  @Override
  public void run() {
    final Queue<Row> queue = QueueFactory.get(queueId);
    try {
      while (isStarted()) {
        if (!queue.isEmpty()) {
          final Row row = queue.poll();
          if (!NULL_ROW.equals(row)) {
            final List<T> values = (List<T>) row.values();
            values.add(this.map(null));
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
  public void stop() {
    this.feed(NULL_ROW);
    super.stop();
  }

  @Override
  public T map(Void element) {
    return supplier.get();
  }

  @Override
  public void setHeader(List<Object> header) {
    super.setHeader(Stream.of(header, List.of(column)).flatMap(List::stream).collect(toList()));
  }
}
