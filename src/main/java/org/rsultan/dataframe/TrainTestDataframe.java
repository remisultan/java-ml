package org.rsultan.dataframe;

import static java.util.Optional.of;

import org.rsultan.dataframe.transform.shuffle.ShuffleDataframe;
import org.rsultan.dataframe.transform.shuffle.ShuffleTransform;
import org.rsultan.dataframe.transform.split.SplitDataframe;
import org.rsultan.dataframe.transform.split.SplitDataframe.TrainTestSplit;
import org.rsultan.dataframe.transform.split.SplitTransform;

public class TrainTestDataframe extends Dataframe implements ShuffleTransform<TrainTestDataframe>,
    SplitTransform {

  private final ShuffleTransform<TrainTestDataframe> shuffleTransform;
  private final SplitDataframe splitTransform;

  private double splitValue = 0.75;

  TrainTestDataframe(Dataframe dataframe) {
    super(dataframe.getColumns());
    this.shuffleTransform = new ShuffleDataframe(this);
    this.splitTransform = new SplitDataframe(this);
  }

  @Override
  public TrainTestDataframe shuffle() {
    return shuffleTransform.shuffle().setSplitValue(splitValue);
  }

  @Override
  public TrainTestSplit split() {
    return splitTransform.split();
  }

  public double getSplitValue() {
    return splitValue;
  }

  public TrainTestDataframe setSplitValue(double splitValue) {
    this.splitValue = of(splitValue)
        .filter(split -> split > 0 && split < 1)
        .orElseThrow(
            () -> new IllegalArgumentException("Split value must be between 0 and 1 excluded")
        );
    return this;
  }
}
