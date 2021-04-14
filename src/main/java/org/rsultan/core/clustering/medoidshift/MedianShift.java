package org.rsultan.core.clustering.medoidshift;

import static org.rsultan.core.clustering.type.MedoidType.MEDIAN;

import org.rsultan.dataframe.Dataframe;

public class MedianShift extends MedoidShift {

  public MedianShift(double bandwidth, long epoch) {
    super(bandwidth, epoch, MEDIAN);
  }

  @Override
  public MedianShift train(Dataframe dataframe) {
    super.train(dataframe);
    return this;
  }
}
