package org.rsultan.dataframe.transform.filter;

import static java.util.stream.Collectors.toList;
import static java.util.stream.IntStream.range;
import static java.util.stream.Stream.of;

import java.util.List;
import java.util.function.BiPredicate;
import java.util.function.Predicate;
import org.rsultan.dataframe.Column;
import org.rsultan.dataframe.Dataframe;
import org.rsultan.dataframe.Dataframes;

public record FilterDataframe(Dataframe dataframe) implements FilterTransform {

  public <SOURCE1> Dataframe filter(String columnName, Predicate<SOURCE1> predicate) {
    List<SOURCE1> values1 = this.dataframe.get(columnName);
    var indices = range(0, values1.size()).parallel()
        .filter(index -> predicate.test(values1.get(index)))
        .boxed().collect(toList());
    return getFilteredDataframe(indices);
  }

  public <SOURCE1, SOURCE2> Dataframe filter(
      String sourceColumn1,
      String sourceColumn2,
      BiPredicate<SOURCE1, SOURCE2> predicate
  ) {
    List<SOURCE1> values1 = this.dataframe.get(sourceColumn1);
    List<SOURCE2> values2 = this.dataframe.get(sourceColumn2);
    var indices = range(0, values1.size()).parallel()
        .filter(index -> predicate.test(values1.get(index), values2.get(index)))
        .boxed().collect(toList());
    return getFilteredDataframe(indices);
  }

  private Dataframe getFilteredDataframe(List<Integer> indices) {
    return Dataframes.create(
        of(this.dataframe.getColumns()).map(column -> new Column<>(column.columnName(),
                range(0, column.values().size()).parallel()
                    .filter(indices::contains)
                    .mapToObj(column.values()::get)
                    .collect(toList())
            )
        ).toArray(Column[]::new)
    );
  }
}
