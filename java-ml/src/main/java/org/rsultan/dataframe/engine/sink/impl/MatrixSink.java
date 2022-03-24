package org.rsultan.dataframe.engine.sink.impl;

import static java.lang.Double.parseDouble;
import static java.util.Optional.ofNullable;
import static java.util.stream.IntStream.range;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.regex.Pattern;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.rsultan.dataframe.Row;
import org.rsultan.dataframe.engine.label.LabelValueIndexer;
import org.rsultan.dataframe.engine.queue.QueueFactory;
import org.rsultan.dataframe.engine.sink.SinkDataProcessor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class MatrixSink extends SinkDataProcessor<INDArray> {

  private final Logger LOG = LoggerFactory.getLogger(MatrixSink.class);
  private static final String NUMBER_REGEX = "^([-+]?(\\d+(\\.\\d+)?|\\.\\d+)([Ee][-+]?\\d+)?)$";
  private static final Pattern PATTERN = Pattern.compile(NUMBER_REGEX);
  private final Map<Object, LabelValueIndexer<?>> columnIndexers;

  private List<INDArray> rows;

  private Integer columnSize;

  public MatrixSink(Map<Object, LabelValueIndexer<?>> columnIndexers) {
    super();
    this.columnIndexers = columnIndexers;
  }

  @Override
  public void consume(Row row) {
    columnSize = ofNullable(columnSize).orElse(header.size());
    var vector = Nd4j.zeros(1, columnSize);
    for (int i = 0; i < columnSize; i++) {
      var key = header.get(i);
      var value = row.get(i);
      if (columnIndexers.containsKey(key)) {
        var index = columnIndexers.get(key).getIndex(value);
        vector.put(0, i, index);
      } else {
        final Double doubleValue = objectToDouble(value);
        vector.put(0, i, doubleValue);
      }
    }
    rows.add(vector);
  }

  private Double objectToDouble(Object obj) {
    if (obj instanceof Number number) {
      return number.doubleValue();
    } else if (obj instanceof Boolean b) {
      return b ? 1.0D : 0.0D;
    } else if (obj instanceof String s && PATTERN.matcher(s).matches()) {
      return parseDouble(s.trim());
    }
    throw new IllegalArgumentException("Cannot cast " + obj + " to number");
  }

  @Override
  public void start(ExecutorService executorService) {
    columnSize = null;
    rows = new ArrayList<>();
    super.start(executorService);
  }

  @Override
  public INDArray getResult() {
    try {
      lock.await();
      return Nd4j.create(rows, rows.size(), columnSize);
    } catch (Exception e) {
      LOG.error("An unexpected error has occurred", e);
      return null;
    }
  }

  public INDArray[] getTrainTest(double threshold) {
    var matrix = getResult();
    int split = (int) (matrix.rows() * threshold);
    return new INDArray[]{
        matrix.getRows(range(0, split).toArray()),
        matrix.getRows(range(split, matrix.rows()).toArray())
    };
  }
}
