package org.rsultan.dataframe.engine;

import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import org.rsultan.dataframe.Row;
import org.rsultan.dataframe.engine.queue.QueueFactory;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.atomic.AtomicBoolean;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static java.util.Optional.ofNullable;

public abstract class BaseDataProcessor implements DataProcessor {

  protected final AtomicBoolean canProcess;
  private static final Logger LOG = LoggerFactory.getLogger(BaseDataProcessor.class);
  protected final String queueId;
  protected List<Object> header;

  protected BaseDataProcessor next;

  protected BaseDataProcessor(String queueId) {
    this.queueId = queueId;
    canProcess = new AtomicBoolean(false);
  }

  public void feed(Row row) {
    ofNullable(next).ifPresent(
        dp -> QueueFactory.get(dp.queueId).add(row)
    );
  }

  public void start(ExecutorService executorService) {
    LOG.debug("Starting {} with queue {}", this.getClass().getSimpleName(), queueId);
    canProcess.set(true);
    executorService.submit(this);
    LOG.debug("{} with queue {} started", this.getClass().getSimpleName(), queueId);
  }

  public void stop() {
    LOG.debug("Stopping {} with queue {}", this.getClass().getSimpleName(), queueId);
    canProcess.set(false);
    QueueFactory.clear(queueId);
    LOG.debug("{} with queue {} stopped", this.getClass().getSimpleName(), queueId);
  }

  public void setNext(BaseDataProcessor next) {
    this.next = next;
  }

  @Override
  public boolean isStarted() {
    return canProcess.get();
  }

  protected AtomicInteger getColumnIndex(List<Object> header, Object sourceName) {
    return new AtomicInteger(header.indexOf(sourceName));
  }

  public void setHeader(List<Object> header) {
    this.header = header;
  }

  public List<Object> getHeader() {
    return header;
  }

  protected void propagateHeader(List<Object> header) {
    var currentStep = next;
    var headersToPropagate = header;
    while (currentStep != null) {
      currentStep.setHeader(headersToPropagate);
      headersToPropagate = currentStep.getHeader();
      currentStep = currentStep.next;
    }
  }
}
