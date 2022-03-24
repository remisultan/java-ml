package org.rsultan.core.clustering.dbscan;

import static java.util.Comparator.comparingInt;
import static java.util.Map.entry;
import static java.util.stream.Collectors.toList;
import static java.util.stream.IntStream.range;
import static org.nd4j.linalg.ops.transforms.Transforms.allEuclideanDistances;
import static org.nd4j.linalg.ops.transforms.Transforms.normalizeZeroMeanAndUnitVariance;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Map.Entry;
import java.util.Set;
import java.util.TreeSet;
import java.util.function.Function;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.rsultan.core.clustering.Clustering;
import org.rsultan.dataframe.Column;
import org.rsultan.dataframe.Dataframe;

public class DBSCAN implements Clustering {

  private final double radius;
  private final int minSamples;

  public  DBSCAN(double radius, int minSamples) {
    this.radius = radius <= 0 ? 1 : radius;
    this.minSamples = minSamples < 1 ? 5 : minSamples;
  }

  @Override
  public DBSCAN train(Dataframe dataframe) {
    throw new IllegalCallerException("Directly predict since this is a clustering algorithm");
  }

  @Override
  public Dataframe predict(Dataframe dataframe) {
    var X = normalizeZeroMeanAndUnitVariance(dataframe.copy().toMatrix());
    var visited = new boolean[X.rows()];
    var clusters = new ArrayList<Set<Integer>>();

    for (int rowIdx = 0; rowIdx < X.rows(); rowIdx++) {
      if (!visited[rowIdx]) {
        var cluster = new TreeSet<Integer>();
        var queue = new LinkedList<Entry<Integer, List<Integer>>>();
        queue.push(entry(rowIdx, getNeighbours(X, rowIdx)));
        while (!queue.isEmpty()) {
          var elt = queue.pop();
          int index = elt.getKey();
          if (!visited[index]) {
            cluster.add(index);
            visited[index] = true;
            var neighbours = elt.getValue();
            if (neighbours.size() >= minSamples) {
              neighbours.stream()
                  .filter(n -> !visited[n])
                  .filter(cluster::add)
                  .map(neighbour -> entry(neighbour, getNeighbours(X, neighbour)))
                  .forEach(queue::add);
            }
          }
        }
        if (!cluster.isEmpty()) {
          clusters.add(cluster);
        }
      }
    }

    var clusterResults = clusters.parallelStream().flatMap(
        l -> l.stream().map(val -> new ClusterResult(clusters.indexOf(l) + 1, val, l.size()))
    ).sorted(comparingInt(ClusterResult::value))
        .collect(toList());

    var clusterNumber = buildColumn(clusterResults, ClusterResult::cluster);
    var clusterDensity = buildColumn(clusterResults, ClusterResult::density);

   return dataframe.copy()
       .addColumn("cluster", clusterNumber)
       .addColumn("density", clusterDensity);
  }

  private List<Integer> buildColumn(List<ClusterResult> clustersResults,
      Function<ClusterResult, Integer> mapper) {
    return clustersResults.parallelStream().map(mapper).collect(toList());
  }

  private List<Integer> getNeighbours(INDArray X, int index) {
    var distances = allEuclideanDistances(X.getRows(index), X, 1);
    return range(0, X.rows()).parallel()
        .filter(idx -> index != idx)
        .filter(idx -> distances.getDouble(idx) <= radius).boxed()
        .collect(toList());
  }

  private static record ClusterResult(int cluster, int value, int density) {

  }
}
