package org.rsultan.dataframe;

import static java.util.stream.Collectors.toList;
import static java.util.stream.IntStream.range;

import java.io.Serializable;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.Executors;
import java.util.function.BiFunction;
import java.util.function.BiPredicate;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.function.Supplier;
import java.util.stream.Stream;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.rsultan.dataframe.engine.BaseDataProcessor;
import org.rsultan.dataframe.engine.filter.RowBiPredicate;
import org.rsultan.dataframe.engine.filter.RowPredicate;
import org.rsultan.dataframe.engine.label.LabelValueIndexer;
import org.rsultan.dataframe.engine.mapper.impl.AddColumnFromList;
import org.rsultan.dataframe.engine.mapper.impl.AddColumnMapper;
import org.rsultan.dataframe.engine.mapper.impl.BiAddColumnMapper;
import org.rsultan.dataframe.engine.mapper.impl.BiColumnTransformer;
import org.rsultan.dataframe.engine.mapper.impl.ColumnTransformer;
import org.rsultan.dataframe.engine.mapper.impl.OneHotEncoderMapper;
import org.rsultan.dataframe.engine.mapper.impl.RemoveColumnMapper;
import org.rsultan.dataframe.engine.mapper.impl.SelectColumnMapper;
import org.rsultan.dataframe.engine.mapper.impl.ShuffleAccumulator;
import org.rsultan.dataframe.engine.mapper.impl.SupplierRowMapper;
import org.rsultan.dataframe.engine.mapper.impl.group.Aggregation;
import org.rsultan.dataframe.engine.mapper.impl.group.AggregationType;
import org.rsultan.dataframe.engine.mapper.impl.group.GroupByTransformer;
import org.rsultan.dataframe.engine.sink.SinkDataProcessor;
import org.rsultan.dataframe.engine.sink.impl.ConsoleSink;
import org.rsultan.dataframe.engine.sink.impl.FileSink;
import org.rsultan.dataframe.engine.sink.impl.MatrixSink;
import org.rsultan.dataframe.engine.sink.impl.RowSink;
import org.rsultan.dataframe.engine.source.SourceDataProcessor;
import org.rsultan.dataframe.transform.FilterTransform;
import org.rsultan.dataframe.transform.MapTransform;
import org.rsultan.dataframe.transform.MatrixTransform;
import org.rsultan.dataframe.transform.ShuffleTransform;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Dataframe implements MapTransform, FilterTransform, MatrixTransform, ShuffleTransform {

  private static final Logger LOG = LoggerFactory.getLogger(Dataframe.class);

  private static record Entry<K, V>(K key, V value) implements Serializable {

  }

  private final List<Entry<Class<? extends BaseDataProcessor>, Object[]>> steps;

  Dataframe(List<Entry<Class<? extends BaseDataProcessor>, Object[]>> steps) {
    this.steps = steps;
  }

  Dataframe(Class<? extends SourceDataProcessor> sourceDataProcessor, Object... args) {
    this(new ArrayList<>());
    addStep(sourceDataProcessor, args);
  }

  public Dataframe select(String... columnNames) {
    addStep(SelectColumnMapper.class, new Object[]{columnNames});
    return this;
  }

  public <SOURCE1> Dataframe filter(String columnName, Predicate<SOURCE1> predicate) {
    addStep(RowPredicate.class, columnName, predicate);
    return this;
  }

  public <SOURCE1, SOURCE2> Dataframe filter(
      String sourceColumn1,
      String sourceColumn2,
      BiPredicate<SOURCE1, SOURCE2> predicate) {
    addStep(RowBiPredicate.class, sourceColumn1, sourceColumn2, predicate);
    return this;
  }

  public <S, T> Dataframe transform(String columnName, Function<S, T> f) {
    addStep(ColumnTransformer.class, columnName, f);
    return this;
  }

  @Override
  public <S1, S2, T> Dataframe transform(String columnName, String columnName2,
      BiFunction<S1, S2, T> f) {
    addStep(BiColumnTransformer.class, columnName, columnName2, f);
    return this;
  }

  @Override
  public <T> Dataframe map(String columnName, Supplier<T> f) {
    addStep(SupplierRowMapper.class, columnName, f);
    return this;
  }

  @Override
  public Dataframe addColumn(Object columnName, List<?> values) {
    addStep(AddColumnFromList.class, columnName, values);
    return this;
  }

  public <S, T> Dataframe map(String columnName, Function<S, T> f, String sourceColumn) {
    addStep(AddColumnMapper.class, columnName, sourceColumn, f);
    return this;
  }

  public <S1, S2, T> Dataframe map(String columnName,
      BiFunction<S1, S2, T> f,
      String sourceColumn1,
      String sourceColumn2) {
    addStep(BiAddColumnMapper.class, columnName, sourceColumn1, sourceColumn2, f);
    return this;
  }

  public Dataframe mapWithout(String... columnNames) {
    Stream.of(columnNames).forEach(col -> addStep(RemoveColumnMapper.class, col));
    return this;
  }

  public static record Result<T>(List<Object> header, T rows) {

  }

  @Override
  public Result<List<Row>> getResult() {
    var sink = new RowSink();
    process(sink);
    final List<Row> result = sink.getResult();
    final List<Object> header = sink.getHeader();
    return new Result<>(header, result);
  }

  @Override
  public <T> List<T> getColumn(Object columnName) {
    var sink = new RowSink();
    process(sink);
    final List<Row> result = sink.getResult();
    var header = sink.getHeader();
    return result.stream().map(row -> {
      var indexOf = header.indexOf(columnName);
      return (T) row.get(indexOf);
    }).collect(toList());
  }

  @Override
  public Dataframe oneHotEncode(String columnToEncode) {
    addStep(OneHotEncoderMapper.class, columnToEncode);
    return this;
  }

  @Override
  public Dataframe shuffle() {
    addStep(ShuffleAccumulator.class);
    return this;
  }

  @Override
  public INDArray toMatrix() {
    return this.toMatrixResult().rows;
  }

  @Override
  public Result<INDArray> toMatrixResult(
      Map<Object, LabelValueIndexer<?>> objectLabelValueIndexerMap) {
    var sink = new MatrixSink(objectLabelValueIndexerMap);
    process(sink);
    var result = sink.getResult();
    final List<Object> header = sink.getHeader();
    return new Result<>(header, result);
  }

  @Override
  public Result<INDArray> toMatrixResult() {
    return toMatrixResult(Map.of());
  }

  @Override
  public INDArray toMatrix(Map<Object, LabelValueIndexer<?>> objectLabelValueIndexerMap) {
    return toMatrixResult(objectLabelValueIndexerMap).rows;
  }

  @Override
  public INDArray[] trainTest(double threshold) {
    var matrix = toMatrix();
    return privateTrainTest(threshold, matrix);
  }

  @Override
  public INDArray[] trainTest(double threshold,
      Map<Object, LabelValueIndexer<?>> labelValueIndexerMap) {
    var matrix = toMatrix(labelValueIndexerMap);
    return privateTrainTest(threshold, matrix);
  }

  private INDArray[] privateTrainTest(double threshold, INDArray matrix) {
    int split = (int) (matrix.rows() * threshold);
    return new INDArray[]{
        matrix.getRows(range(0, split).toArray()),
        matrix.getRows(range(split, matrix.rows()).toArray())
    };
  }

  public void show(int number) {
    this.show(0, number);
  }

  public void show(int start, int end) {
    process(new ConsoleSink(start, end));
  }

  public void write(String filename, String separator, String enclosure) {
    process(new FileSink(filename, separator, enclosure, true));
  }

  void addStep(Class<? extends BaseDataProcessor> clazz, Object... args) {
    steps.add(new Entry<>(clazz, args));
  }

  public Dataframe groupBy(String source, String target, AggregationType aggregationType) {
    return this.groupBy(source, new Aggregation(target, aggregationType));
  }

  public Dataframe groupBy(String source, Aggregation... aggregations) {
    addStep(GroupByTransformer.class, source, aggregations);
    return this;
  }

  private synchronized void process(SinkDataProcessor<?> sink) {
    if (steps.size() > 0) {
      var dataProcessors = new ArrayList<BaseDataProcessor>(steps.size() + 1);
      BaseDataProcessor previous = null;
      for (var step : steps) {
        var dataProcessor = createDataProcessor(step).orElseThrow(
            () -> new IllegalArgumentException(
                "Could not create data processor: " + step.key().getSimpleName())
        );
        if (previous != null) {
          previous.setNext(dataProcessor);
        }
        dataProcessors.add(dataProcessor);
        previous = dataProcessor;
      }
      previous.setNext(sink);
      dataProcessors.add(sink);
      var executor = Executors.newFixedThreadPool(dataProcessors.size());
      dataProcessors.forEach(dp -> dp.start(executor));
      executor.shutdown();
    }
  }

  private Optional<BaseDataProcessor> createDataProcessor(
      Entry<Class<? extends BaseDataProcessor>, Object[]> e) {
    try {
      var constructor = (Constructor<BaseDataProcessor>) e.key().getConstructors()[0];
      return Optional.of(constructor.newInstance(e.value()));
    } catch (InstantiationException | IllegalAccessException | InvocationTargetException exc) {
      LOG.error("Error creating data processor", exc.getCause());
      return Optional.empty();
    }
  }

  public Dataframe copy() {
    return new Dataframe(new ArrayList<>(steps));
  }

}



