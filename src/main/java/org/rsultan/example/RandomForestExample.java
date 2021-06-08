package org.rsultan.example;

import static java.lang.Boolean.parseBoolean;
import static org.rsultan.core.tree.impurity.ImpurityStrategy.ENTROPY;

import java.io.File;
import java.io.IOException;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;
import org.rsultan.core.Models;
import org.rsultan.core.ensemble.rf.RandomForestClassifier;
import org.rsultan.core.ensemble.rf.RandomForestRegressor;
import org.rsultan.dataframe.Dataframes;

public class RandomForestExample {

  /*
    You can use the iris dataset for classification --> args[0]
    You can use the wine dataset for regression --> args[0]
   */

  public static final String TEMP_DIR = System.getProperty("java.io.tmpdir");

  static {
    Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);
  }

  public static void main(String[] args) throws IOException, ClassNotFoundException {
    boolean isClassifier = parseBoolean(args[1]);
    if (isClassifier) {
      classifier(args[0]);
    } else {
      regressor(args[0]);
    }
  }

  private static void regressor(String arg) throws IOException, ClassNotFoundException {
    File file = new File(TEMP_DIR + File.separator + "rfRegressor.gz");
    String pathName = file.toPath().toString();
    RandomForestRegressor randomForestRegression;
    var dataframe = Dataframes.csvTrainTest(arg, ";").shuffle();
    var dfSplit = dataframe.setSplitValue(0.5).split();
    if (!file.exists()) {
      randomForestRegression = new RandomForestRegressor(10)
          .setTreeDepth(5)
          .setSampleSizeRatio(0.4)
          .setResponseVariableName("quality")
          .train(dfSplit.train());
      Models.write(pathName, randomForestRegression);
    } else {
      randomForestRegression = Models.read(pathName);
    }
    var newDf = randomForestRegression.predict(dfSplit.test());
    newDf.show(0, 15000);
  }

  private static void classifier(String arg) throws IOException, ClassNotFoundException {
    File file = new File(TEMP_DIR + File.separator + "rfClassifier.gz");
    RandomForestClassifier randomForestClassifier;
    var dataframe = Dataframes.csvTrainTest(arg, ",", "\"", false).shuffle();
    var dfSplit = dataframe.setSplitValue(0.4).split();
    if (!file.exists()) {
      randomForestClassifier = new RandomForestClassifier(100, ENTROPY)
          .setTreeDepth(5)
          .setSampleSizeRatio(0.4)
          .setResponseVariableName("c4")
          .train(dfSplit.train());
      Models.write(file.toPath(), randomForestClassifier);
    } else {
      randomForestClassifier = Models.read(file.toPath());
    }

    var newDf = randomForestClassifier.predict(dfSplit.test());
    newDf.show(0, 150);
  }
}
