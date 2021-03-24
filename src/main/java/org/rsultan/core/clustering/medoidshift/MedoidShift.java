package org.rsultan.core.clustering.medoidshift;


import static java.util.stream.Collectors.toList;
import static java.util.stream.LongStream.range;

import java.util.Arrays;
import java.util.concurrent.atomic.AtomicBoolean;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.rsultan.core.clustering.Clustering;
import org.rsultan.core.clustering.type.MedoidType;
import org.rsultan.dataframe.Column;
import org.rsultan.dataframe.Dataframe;

public abstract class MedoidShift implements Clustering {

  private final MedoidType medoidType;
  private final long bandwidth;
  private final long epoch;

  private INDArray C;
  private INDArray Xt;

  protected MedoidShift(long bandwidth, long epoch, MedoidType medoidType) {
    this.medoidType = medoidType;
    this.epoch = epoch;
    this.bandwidth = bandwidth;
  }

  @Override
  public MedoidShift train(Dataframe dataframe) {
    var medoidFactory = medoidType.getMedoidFactory();
    var isTerminated = new AtomicBoolean(false);
    Xt = dataframe.toMatrix().transpose();
    C = Xt.dup();
    range(0, epoch)
        .filter(epoch -> !isTerminated.get())
        .forEach(epoch -> {
          System.out.println("Epoch " + epoch);
          var newCentroidList = range(0, C.columns())
              .parallel().unordered()
              .mapToObj(col -> {
                var centroid = C.getColumn(col);
                var range = epoch == 0 ? range(col, Xt.columns()) : range(0, Xt.columns());
                return range.parallel().unordered().mapToObj(Xt::getColumn)
                    .filter(feature -> medoidFactory.computeNorm(feature.sub(centroid)) < bandwidth)
                    .collect(toList());
              })
              .map(features -> Nd4j.create(features, features.size(), C.rows()))
              .map(medoidFactory::computeMedoids)
              .distinct()
              .collect(toList());
          var newC = Nd4j.create(newCentroidList, newCentroidList.size(), C.rows())
              .transpose();
          if (C.equalShapes(newC) && C.equals(newC)) {
            isTerminated.set(true);
          }
          C = newC;
          System.out.println(C);
          System.out.println(Arrays.toString(C.shape()));
        });

    return this;
  }

  @Override
  public Dataframe predict(Dataframe dataframe) {
    var medoidFactory = medoidType.getMedoidFactory();
    var Xpredict = dataframe.toMatrix();
    var distances = medoidFactory.computeDistance(Xpredict, C.transpose());
    var indices = Nd4j.argMin(distances, 1);
    var predictions = C.transpose().get(indices);
    return dataframe.addColumn(new Column<>("predictions",
        range(0, predictions.rows()).mapToObj(predictions::getRow).collect(toList())));
  }

  public INDArray getC() {
    return C;
  }
}
