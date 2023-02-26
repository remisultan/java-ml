package org.rsultan.dataframe.engine.mapper.impl.group;

import static java.util.Comparator.comparing;
import static java.util.stream.Collectors.groupingBy;
import static java.util.stream.Collectors.toList;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.TreeMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.apache.commons.lang3.NotImplementedException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.ops.NDLinalg;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.rsultan.dataframe.Row;
import org.rsultan.dataframe.engine.mapper.AccumulatorDataProcessor;
import org.rsultan.dataframe.engine.mapper.impl.accumulator.Accumulator;
import org.rsultan.dataframe.engine.mapper.impl.accumulator.Average;
import org.rsultan.dataframe.engine.mapper.impl.accumulator.Count;
import org.rsultan.dataframe.engine.mapper.impl.accumulator.ListAccumulator;
import org.rsultan.dataframe.engine.mapper.impl.accumulator.Max;
import org.rsultan.dataframe.engine.mapper.impl.accumulator.Min;
import org.rsultan.dataframe.engine.mapper.impl.accumulator.Sum;

/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public class GroupByTransformer extends AccumulatorDataProcessor {

  private final Aggregation[] aggregations;
  private final Map<AggregationKey, Accumulator<?, ?>> accumulators = new HashMap<>();
  private final Object sourceColumn;

  public GroupByTransformer(Object sourceColumn, Aggregation... aggregations) {
    this.sourceColumn = sourceColumn;
    this.aggregations = aggregations;
  }

  @Override
  public Row map(Row element) {
    throw new NotImplementedException("Not Implemented");
  }

  @Override
  protected void accumulate(Row row) {
    for (Aggregation agg : aggregations) {
      var source = row.get(getColumnIndex(header, sourceColumn).get());
      var target = row.get(getColumnIndex(header, agg.target()).get());
      getAccumulator(source, agg).accumulate(target);
    }
  }

  private Accumulator getAccumulator(Object source, Aggregation agg) {
    final AggregationKey key = new AggregationKey(source, agg);
    accumulators.putIfAbsent(key, buildAccumulator(agg));
    return accumulators.get(key);
  }

  private Accumulator<?, ?> buildAccumulator(Aggregation agg) {
    return switch (agg.type()) {
      case COUNT -> new Count();
      case SUM -> new Sum();
      case AVG -> new Average();
      case MIN -> new Min();
      case MAX -> new Max();
      case ACCUMULATE -> new ListAccumulator();
    };
  }

  @Override
  protected void feedFromAccumulator() {
    List<Object> header = Stream.of(aggregations).map(Aggregation::getColumnName).collect(toList());
    header.add(0, sourceColumn);
    super.propagateHeader(header);

    var grouped = this.accumulators.entrySet().stream()
        .collect(groupingBy(e -> e.getKey().source()));
    grouped.entrySet().stream().map(e -> {
      var data = new ArrayList<>();

      var source = e.getKey();
      var aggKey = e.getValue();
      data.add(source);

      Stream.of(this.aggregations).map(Aggregation::type).map(type -> {
        var value = aggKey.stream().filter(agg -> type.equals(agg.getKey().aggregation().type()))
            .map(Entry::getValue)
            .findFirst();
        return value.map(Accumulator::get).orElse(null);
      }).forEach(data::add);

      return new Row(data);
    }).forEach(this::feed);
  }
}
