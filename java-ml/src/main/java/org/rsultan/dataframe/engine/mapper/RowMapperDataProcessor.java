package org.rsultan.dataframe.engine.mapper;

import static org.rsultan.utils.Constants.NULL_ROW;

import org.rsultan.dataframe.Row;
import org.rsultan.dataframe.engine.BaseDataProcessor;
import org.rsultan.dataframe.engine.queue.QueueFactory;

import java.util.Queue;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public abstract class RowMapperDataProcessor extends BaseDataProcessor implements Mapper<Row, Row> {

  private static final Logger LOGGER = LoggerFactory.getLogger(RowMapperDataProcessor.class);

  protected RowMapperDataProcessor() {
    super(QueueFactory.create());
  }

  @Override
  public void run() {
    try {
      while (isStarted()) {
      final Queue<Row> queue = QueueFactory.get(queueId);
        if (!queue.isEmpty()) {
          final Row row = queue.poll();
          if (!NULL_ROW.equals(row)) {
            this.feed(this.map(row));
          } else {
            stop();
          }
        }
      }
    } catch (Exception e) {
      LOGGER.error("Error in RowMapperDataProcessor", e);
      stop();
    }
  }

  @Override
  public void stop() {
    this.feed(NULL_ROW);
    super.stop();
  }
}
