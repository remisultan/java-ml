package org.rsultan.example;

import static java.lang.Boolean.parseBoolean;
import static org.rsultan.core.tree.impurity.ImpurityStrategy.GINI;

import java.io.File;
import java.io.IOException;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;
import org.rsultan.core.Models;
import org.rsultan.core.tree.DecisionTreeClassifier;
import org.rsultan.core.tree.DecisionTreeRegressor;
import org.rsultan.dataframe.Dataframes;

public class DecisionTreeClassifierExample {

  public static final String TEMP_DIR = System.getProperty("java.io.tmpdir");

  /*
    You can use the iris dataset for classification --> args[0]
    You can use the wine dataset for regression --> args[0]
   */

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
    File file = new File(TEMP_DIR + File.separator + "dtRegressor.gz");
    var pathName = file.toPath();
    DecisionTreeRegressor decisionTreeRegression;
    var dataframe = Dataframes.csvTrainTest(arg, ";").shuffle();
    var dfSplit = dataframe.setSplitValue(0.5).split();
    if (!file.exists()) {
      decisionTreeRegression = new DecisionTreeRegressor(5)
          .setResponseVariableName("alcohol")
          .train(dfSplit.train());
    } else {
      decisionTreeRegression = Models.read(pathName);
    }
    var newDf = decisionTreeRegression.predict(dfSplit.test());
    newDf.show(0, 15000);
  }

  private static void classifier(String arg) throws IOException, ClassNotFoundException {
    File file = new File(TEMP_DIR + File.separator + "dtClassifier.gz");
    var pathName = file.toPath();
    DecisionTreeClassifier decisionTreeClassifier;
    var dataframe = Dataframes.csvTrainTest(arg, ",", "\"", false).shuffle();
    var dfSplit = dataframe.setSplitValue(0.4).split();
    if (!file.exists()) {
      decisionTreeClassifier = new DecisionTreeClassifier(5, GINI);
      decisionTreeClassifier.setResponseVariableName("c4").train(dfSplit.train());
      Models.write(pathName, decisionTreeClassifier);
    } else {
      decisionTreeClassifier = Models.read(pathName);
    }
    var newDf = decisionTreeClassifier.predict(dfSplit.test());
    newDf.show(0, 150);
  }
}
