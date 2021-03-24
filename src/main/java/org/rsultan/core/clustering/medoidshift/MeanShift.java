package org.rsultan.core.clustering.medoidshift;

import static org.rsultan.core.clustering.type.MedoidType.MEAN;

import org.rsultan.dataframe.Dataframe;

public class MeanShift extends MedoidShift {

  public MeanShift(long bandwidth, long epoch) {
    super(bandwidth, epoch, MEAN);
  }

  @Override
  public MeanShift train(Dataframe dataframe) {
    super.train(dataframe);
    return this;
  }
}
