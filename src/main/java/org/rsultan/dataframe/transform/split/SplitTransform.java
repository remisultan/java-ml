package org.rsultan.dataframe.transform.split;

import org.rsultan.dataframe.transform.split.SplitDataframe.TrainTestSplit;

public interface SplitTransform {

  TrainTestSplit split();

}
